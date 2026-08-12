#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using RC = std::array<int,2>;
static constexpr int DR[5]={0,-1,1,0,0}, DC[5]={0,0,0,-1,1};
static constexpr int INF=std::numeric_limits<int>::max();

class EpisodeFeatureGenerator {
 public:
  EpisodeFeatureGenerator(
      py::array_t<uint8_t,py::array::c_style|py::array::forcecast> obstacles,
      py::array_t<int64_t,py::array::c_style|py::array::forcecast> positions,
      py::array_t<int64_t,py::array::c_style|py::array::forcecast> goals,
      py::array_t<int64_t,py::array::c_style|py::array::forcecast> actions,
      int local=17,int core=15,int max_current=14,int tracks=14,int history=2,
      int message_neighbors=6,int max_hops=1023)
      :local_(local),core_(core),max_current_(max_current),tracks_(tracks),history_(history),
       message_neighbors_(message_neighbors),max_hops_(max_hops) {
    auto o=obstacles.request(),p=positions.request(),g=goals.request(),a=actions.request();
    if(o.ndim!=2||p.ndim!=3||p.shape[2]!=2||a.ndim!=2||g.ndim<2||g.ndim>3||g.shape[g.ndim-1]!=2)
      throw std::invalid_argument("invalid episode array shape");
    h_=o.shape[0];w_=o.shape[1];tp_=p.shape[0];n_=p.shape[1];ta_=a.shape[0]; dynamic_goals_=g.ndim==3;
    if(a.shape[1]!=n_||(dynamic_goals_?(g.shape[0]!=tp_||g.shape[1]!=n_):g.shape[0]!=n_)) throw std::invalid_argument("agent/time dimensions differ");
    obstacles_.assign((uint8_t*)o.ptr,(uint8_t*)o.ptr+h_*w_);
    auto *pp=(int64_t*)p.ptr; positions_.resize(tp_*n_);
    for(int z=0;z<tp_*n_;++z) positions_[z]={(int)pp[2*z],(int)pp[2*z+1]};
    auto *gp=(int64_t*)g.ptr; int gn=(dynamic_goals_?tp_*n_:n_); goals_.resize(gn);
    for(int z=0;z<gn;++z) goals_[z]={(int)gp[2*z],(int)gp[2*z+1]};
    auto *ap=(int64_t*)a.ptr; actions_.resize(ta_*n_); for(int z=0;z<ta_*n_;++z) actions_[z]=(int)ap[z];
    degree_.resize(h_*w_); for(int r=0;r<h_;++r)for(int c=0;c<w_;++c) degree_[r*w_+c]=degree({r,c});
    if(!dynamic_goals_){ distances_.resize(n_); for(int i=0;i<n_;++i)distances_[i]=distance_map(goal(0,i)); }
  }

  void update_history(
      py::array_t<int64_t,py::array::c_style|py::array::forcecast> positions,
      py::array_t<int64_t,py::array::c_style|py::array::forcecast> actions) {
    auto p=positions.request(),a=actions.request();
    if(p.ndim!=3||p.shape[1]!=n_||p.shape[2]!=2||a.ndim!=2||a.shape[1]!=n_)
      throw std::invalid_argument("invalid online history shape");
    tp_=p.shape[0];ta_=a.shape[0];auto*pp=(int64_t*)p.ptr;positions_.resize(tp_*n_);
    for(int z=0;z<tp_*n_;++z)positions_[z]={(int)pp[2*z],(int)pp[2*z+1]};
    auto*ap=(int64_t*)a.ptr;actions_.resize(ta_*n_);for(int z=0;z<ta_*n_;++z)actions_[z]=(int)ap[z];
  }

  py::dict build(int step) {
    if(step<0||step>=ta_) throw std::out_of_range("step outside actions");
    if(dynamic_goals_){ distances_.resize(n_); for(int i=0;i<n_;++i)distances_[i]=distance_map(goal(step,i)); }
    const int B=n_,N=max_current_,A=5,H=tracks_,T=history_,center=core_/2;
    py::array_t<int64_t> lm({B,local_,local_}),cxy({B,N,2}),cgoal({B,N,2}),chops({B,N}),cgids({B,N});
    py::array_t<bool> cvalid({B,N}),creset({B,N});
    py::array_t<int64_t> target({B,N,A,2}),onehop({B,N,A}),delta({B,N,A});
    py::array_t<bool> incore({B,N,A}),free({B,N,A}),greedy({B,N,A}),bneck({B,N,A}),occupied({B,N,A});
    py::array_t<int64_t> hxy({B,H,T,2}),hgoal({B,H,T,2}),hhops({B,H,T}),hsel({B,H,T}),hobs({B,H,T}),hcslot({B,H}),hgids({B,H});
    py::array_t<bool> hvalid({B,H,T}); py::array_t<int64_t> egoaction({B});
    py::array_t<int64_t> nindex({B,message_neighbors_}),nslot({B,message_neighbors_}); py::array_t<bool> nvalid({B,message_neighbors_});
    auto LM=lm.mutable_unchecked<3>();auto CXY=cxy.mutable_unchecked<3>();auto CG=cgoal.mutable_unchecked<3>();auto CH=chops.mutable_unchecked<2>();auto CGI=cgids.mutable_unchecked<2>();
    auto CV=cvalid.mutable_unchecked<2>();auto CR=creset.mutable_unchecked<2>();auto TG=target.mutable_unchecked<4>();auto OH=onehop.mutable_unchecked<3>();auto DE=delta.mutable_unchecked<3>();
    auto IC=incore.mutable_unchecked<3>();auto FR=free.mutable_unchecked<3>();auto GR=greedy.mutable_unchecked<3>();auto BN=bneck.mutable_unchecked<3>();auto OC=occupied.mutable_unchecked<3>();
    auto HX=hxy.mutable_unchecked<4>();auto HG=hgoal.mutable_unchecked<4>();auto HH=hhops.mutable_unchecked<3>();auto HS=hsel.mutable_unchecked<3>();auto HO=hobs.mutable_unchecked<3>();auto HV=hvalid.mutable_unchecked<3>();auto HC=hcslot.mutable_unchecked<2>();auto HI=hgids.mutable_unchecked<2>();
    auto EA=egoaction.mutable_unchecked<1>();auto NI=nindex.mutable_unchecked<2>();auto NS=nslot.mutable_unchecked<2>();auto NV=nvalid.mutable_unchecked<2>();

    for(int ego=0;ego<B;++ego){
      auto current=rank_current(step,ego); auto history=history_ids(step,ego,current);
      // History ranking may inspect time-varying goals at earlier lags.
      // Restore current-step maps before emitting the current candidate fields.
      if(dynamic_goals_) for(int i=0;i<n_;++i) distances_[i]=distance_map(goal(step,i));
      std::unordered_map<int,int> slot; for(int s=0;s<(int)current.size();++s)slot[current[s]]=s;
      std::set<int> previous; if(step>0){auto p=rank_current(step-1,ego);previous.insert(p.begin(),p.end());}else previous.insert(ego);
      RC ep=pos(step,ego);
      for(int r=0;r<local_;++r)for(int c=0;c<local_;++c){int rr=ep[0]-local_/2+r,cc=ep[1]-local_/2+c;LM(ego,r,c)=inside(rr,cc)?obstacles_[rr*w_+cc]:1;}
      for(int s=0;s<N;++s){
        bool ok=s<(int)current.size();CV(ego,s)=ok;CR(ego,s)=false;CGI(ego,s)=-1;CXY(ego,s,0)=CXY(ego,s,1)=0;CG(ego,s,0)=CG(ego,s,1)=0;CH(ego,s)=max_hops_+2;
        for(int ac=0;ac<A;++ac){TG(ego,s,ac,0)=TG(ego,s,ac,1)=0;IC(ego,s,ac)=FR(ego,s,ac)=GR(ego,s,ac)=BN(ego,s,ac)=OC(ego,s,ac)=false;OH(ego,s,ac)=max_hops_+1;DE(ego,s,ac)=5;}
        if(!ok)continue;int gid=current[s];RC p=pos(step,gid),g=goal(step,gid);CGI(ego,s)=gid;CR(ego,s)=!previous.count(gid);CXY(ego,s,0)=p[0]-ep[0];CXY(ego,s,1)=p[1]-ep[1];CG(ego,s,0)=g[0]-p[0];CG(ego,s,1)=g[1]-p[1];
        int cd=distances_[gid][index(p)];CH(ego,s)=cd==INF?-1:cd;
        for(int ac=0;ac<A;++ac){RC q={p[0]+DR[ac],p[1]+DC[ac]};int rr=p[0]-ep[0]+DR[ac]+center,cc=p[1]-ep[1]+DC[ac]+center;TG(ego,s,ac,0)=rr;TG(ego,s,ac,1)=cc;IC(ego,s,ac)=rr>=0&&rr<core_&&cc>=0&&cc<core_;for(int other=0;other<n_;++other)if(other!=gid&&pos(step,other)==q){OC(ego,s,ac)=true;break;}bool f=inside(q[0],q[1])&&obstacles_[index(q)]==0;FR(ego,s,ac)=f;if(!f){DE(ego,s,ac)=3;continue;}int td=distances_[gid][index(q)];OH(ego,s,ac)=td==INF?-1:td;if(cd==INF||td==INF)DE(ego,s,ac)=4;else if(td<cd){DE(ego,s,ac)=0;GR(ego,s,ac)=true;}else if(td>cd)DE(ego,s,ac)=2;else DE(ego,s,ac)=1;BN(ego,s,ac)=degree_[index(q)]<=2;}if(CH(ego,s)==0)GR(ego,s,0)=true;
      }
      for(int tr=0;tr<H;++tr){HC(ego,tr)=-1;HI(ego,tr)=-1;for(int z=0;z<T;++z){HX(ego,tr,z,0)=HX(ego,tr,z,1)=0;HG(ego,tr,z,0)=HG(ego,tr,z,1)=0;HH(ego,tr,z)=max_hops_+2;HS(ego,tr,z)=HO(ego,tr,z)=5;HV(ego,tr,z)=false;}if(tr>=(int)history.size())continue;int gid=history[tr];HI(ego,tr)=gid;auto it=slot.find(gid);HC(ego,tr)=it==slot.end()?-1:it->second;for(int lag=1;lag<=T;++lag){int tau=step-lag;if(tau<0||!is_visible(tau,ego,gid))continue;HV(ego,tr,lag-1)=true;RC p=pos(tau,gid),g=goal(tau,gid);HX(ego,tr,lag-1,0)=p[0]-ep[0];HX(ego,tr,lag-1,1)=p[1]-ep[1];HG(ego,tr,lag-1,0)=g[0]-p[0];HG(ego,tr,lag-1,1)=g[1]-p[1];int d=(dynamic_goals_?distance_map(g):distances_[gid])[index(p)];HH(ego,tr,lag-1)=d==INF?-1:d;HS(ego,tr,lag-1)=actions_[tau*n_+gid];HO(ego,tr,lag-1)=action_delta(pos(tau+1,gid),p);}}
      EA(ego)=actions_[step*n_+ego];for(int k=0;k<message_neighbors_;++k){NI(ego,k)=0;NS(ego,k)=max_current_;NV(ego,k)=false;}int k=0;for(int tr=1;tr<H&&k<message_neighbors_;++tr){int gid=HI(ego,tr),sl=HC(ego,tr);if(gid<0||sl<0||gid>=B)continue;NI(ego,k)=gid;NS(ego,k)=sl;NV(ego,k)=true;++k;}
    }
    py::dict out;out["local_maps"]=lm;out["current_xy"]=cxy;out["current_goal_delta"]=cgoal;out["current_hops"]=chops;out["current_valid"]=cvalid;out["current_track_reset"]=creset;out["current_global_ids"]=cgids;out["candidate_target_core_xy"]=target;out["candidate_in_core"]=incore;out["candidate_static_free"]=free;out["candidate_one_hop_hops"]=onehop;out["candidate_delta_ctg"]=delta;out["candidate_greedy"]=greedy;out["candidate_bottleneck"]=bneck;out["candidate_dynamic_occupied"]=occupied;out["history_xy"]=hxy;out["history_goal_delta"]=hgoal;out["history_hops"]=hhops;out["history_selected_action"]=hsel;out["history_observed_move"]=hobs;out["history_valid"]=hvalid;out["history_track_current_slot"]=hcslot;out["history_global_ids"]=hgids;out["ego_action"]=egoaction;out["neighbor_view_index"]=nindex;out["neighbor_valid"]=nvalid;out["neighbor_current_slot"]=nslot;return out;
  }

 private:
  int h_,w_,tp_,ta_,n_,local_,core_,max_current_,tracks_,history_,message_neighbors_,max_hops_;bool dynamic_goals_;
  std::vector<uint8_t>obstacles_;std::vector<RC>positions_,goals_;std::vector<int>actions_,degree_;std::vector<std::vector<int>>distances_;
  bool inside(int r,int c)const{return r>=0&&r<h_&&c>=0&&c<w_;}int index(const RC&p)const{return p[0]*w_+p[1];}RC pos(int t,int i)const{return positions_[t*n_+i];}RC goal(int t,int i)const{return goals_[dynamic_goals_?t*n_+i:i];}
  int degree(const RC&p)const{int d=0;for(int a=1;a<5;++a){int r=p[0]+DR[a],c=p[1]+DC[a];if(inside(r,c)&&obstacles_[r*w_+c]==0)++d;}return d;}
  std::vector<int>distance_map(const RC&g)const{std::vector<int>d(h_*w_,INF);if(!inside(g[0],g[1])||obstacles_[index(g)])return d;std::queue<RC>q;q.push(g);d[index(g)]=0;while(!q.empty()){RC p=q.front();q.pop();for(int a=1;a<5;++a){RC x={p[0]+DR[a],p[1]+DC[a]};if(inside(x[0],x[1])&&!obstacles_[index(x)]&&d[index(x)]==INF){d[index(x)]=d[index(p)]+1;q.push(x);}}}return d;}
  bool is_visible(int t,int ego,int gid)const{RC e=pos(t,ego),p=pos(t,gid);int r=core_/2;return std::abs(e[0]-p[0])<=r&&std::abs(e[1]-p[1])<=r;}
  std::pair<std::set<RC>,std::set<RC>>candidate(int t,int gid){RC p=pos(t,gid),g=goal(t,gid);auto &d=distances_[gid];if(dynamic_goals_)d=distance_map(g);int cd=d[index(p)];std::set<RC>f,gr;for(int a=0;a<5;++a){RC q={p[0]+DR[a],p[1]+DC[a]};if(!inside(q[0],q[1])||obstacles_[index(q)])continue;f.insert(q);if(cd!=INF&&d[index(q)]<cd)gr.insert(q);}if(p==g)gr.insert(p);return{f,gr};}
  static int overlap(const std::set<RC>&a,const std::set<RC>&b){int n=0;for(auto&x:a)n+=b.count(x);return n;}
  std::vector<int>rank_current(int t,int ego){std::vector<int>v;for(int i=0;i<n_;++i)if(i!=ego&&is_visible(t,ego,i))v.push_back(i);RC ep=pos(t,ego);std::sort(v.begin(),v.end(),[&](int a,int b){RC pa=pos(t,a),pb=pos(t,b);int da=std::abs(pa[0]-ep[0])+std::abs(pa[1]-ep[1]),db=std::abs(pb[0]-ep[0])+std::abs(pb[1]-ep[1]);return da!=db?da<db:a<b;});v.insert(v.begin(),ego);if((int)v.size()>max_current_)v.resize(max_current_);return v;}
  std::vector<int>history_ids(int,const int,const std::vector<int>&current){std::vector<int>out=current;if((int)out.size()>tracks_)out.resize(tracks_);return out;}
  static int action_delta(const RC&a,const RC&b){int dr=a[0]-b[0],dc=a[1]-b[1];for(int x=0;x<5;++x)if(DR[x]==dr&&DC[x]==dc)return x;return 6;}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){py::class_<EpisodeFeatureGenerator>(m,"EpisodeFeatureGenerator").def(py::init<py::array_t<uint8_t,py::array::c_style|py::array::forcecast>,py::array_t<int64_t,py::array::c_style|py::array::forcecast>,py::array_t<int64_t,py::array::c_style|py::array::forcecast>,py::array_t<int64_t,py::array::c_style|py::array::forcecast>,int,int,int,int,int,int,int>()).def("update_history",&EpisodeFeatureGenerator::update_history).def("build",&EpisodeFeatureGenerator::build);}

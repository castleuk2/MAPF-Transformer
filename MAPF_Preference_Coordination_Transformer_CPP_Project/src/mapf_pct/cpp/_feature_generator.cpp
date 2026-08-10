#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using RC = std::array<int, 2>;
static constexpr int DR[5] = {0, -1, 1, 0, 0};
static constexpr int DC[5] = {0, 0, 0, -1, 1};
static constexpr int INF = std::numeric_limits<int>::max();

class StatefulFeatureGenerator {
 public:
  StatefulFeatureGenerator(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> obstacles,
                           py::array_t<int64_t, py::array::c_style | py::array::forcecast> goals,
                           int core_size = 15, int local_size = 17, int max_agents = 25,
                           int history_steps = 5, int max_hops = 1023, int goal_clip = 31,
                           int contender_buckets = 8)
      : core_(core_size), local_(local_size), max_agents_(max_agents), history_steps_(history_steps),
        max_hops_(max_hops), goal_clip_(goal_clip), contender_buckets_(contender_buckets) {
    auto om = obstacles.request(); auto gm = goals.request();
    if (om.ndim != 2 || gm.ndim != 2 || gm.shape[1] != 2) throw std::invalid_argument("invalid map/goals shape");
    h_ = static_cast<int>(om.shape[0]); w_ = static_cast<int>(om.shape[1]); n_ = static_cast<int>(gm.shape[0]);
    obstacles_.assign(static_cast<uint8_t*>(om.ptr), static_cast<uint8_t*>(om.ptr) + h_ * w_);
    auto *gp = static_cast<int64_t*>(gm.ptr); goals_.resize(n_);
    for (int i=0;i<n_;++i) goals_[i] = {static_cast<int>(gp[2*i]), static_cast<int>(gp[2*i+1])};
    distances_.reserve(n_); for (const auto &g : goals_) distances_.push_back(distance_map(g));
  }

  void reset() { positions_history_.clear(); selected_history_.clear(); }

  void commit_actions(py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions) {
    auto a=actions.request(); if (a.ndim!=1 || a.shape[0]!=n_) throw std::invalid_argument("actions must be [N]");
    auto *p=static_cast<int64_t*>(a.ptr); std::vector<int> row(n_);
    for(int i=0;i<n_;++i) row[i]=std::clamp(static_cast<int>(p[i]),0,4);
    selected_history_.push_back(std::move(row));
  }

  py::dict generate(py::array_t<int64_t, py::array::c_style | py::array::forcecast> positions) {
    auto pm=positions.request(); if(pm.ndim!=2 || pm.shape[0]!=n_ || pm.shape[1]!=2) throw std::invalid_argument("positions must be [N,2]");
    auto *pp=static_cast<int64_t*>(pm.ptr); std::vector<RC> current(n_);
    for(int i=0;i<n_;++i) current[i]={static_cast<int>(pp[2*i]),static_cast<int>(pp[2*i+1])};
    if (positions_history_.size() != selected_history_.size()) {
      throw std::runtime_error("generate must be called once before each commit_actions");
    }
    positions_history_.push_back(current);

    const int B=n_, M=max_agents_, A=5, H=history_steps_, radius=core_/2;
    py::array_t<int64_t> local_maps({B,local_,local_});
    py::array_t<int64_t> agent_xy({B,M,2}), goal_delta({B,M,2}), remaining({B,M});
    py::array_t<bool> valid({B,M}), on_goal({B,M}), goal_outside({B,M});
    py::array_t<int64_t> hsel({B,M,H}), hexec({B,M,H}), hout({B,M,H}), hdelta({B,M,H});
    py::array_t<bool> hvalid({B,M,H});
    py::array_t<int64_t> target_xy({B,M,A,2}), cdelta({B,M,A}), contenders({B,M,A});
    py::array_t<bool> in_view({B,M,A}), greedy({B,M,A}), static_free({B,M,A}), occupied({B,M,A}), edge_swap({B,M,A}), bottleneck({B,M,A});
    py::array_t<float> congestion({B,M,A});
    py::array_t<int64_t> slot_ids({B,M});

    auto lm=local_maps.mutable_unchecked<3>(); auto xy=agent_xy.mutable_unchecked<3>(); auto gd=goal_delta.mutable_unchecked<3>();
    auto rem=remaining.mutable_unchecked<2>(); auto va=valid.mutable_unchecked<2>(); auto og=on_goal.mutable_unchecked<2>(); auto go=goal_outside.mutable_unchecked<2>();
    auto hs=hsel.mutable_unchecked<3>(); auto he=hexec.mutable_unchecked<3>(); auto ho=hout.mutable_unchecked<3>(); auto hd=hdelta.mutable_unchecked<3>(); auto hv=hvalid.mutable_unchecked<3>();
    auto tx=target_xy.mutable_unchecked<4>(); auto cd=cdelta.mutable_unchecked<3>(); auto ct=contenders.mutable_unchecked<3>();
    auto iv=in_view.mutable_unchecked<3>(); auto gr=greedy.mutable_unchecked<3>(); auto sf=static_free.mutable_unchecked<3>(); auto oc=occupied.mutable_unchecked<3>();
    auto es=edge_swap.mutable_unchecked<3>(); auto bn=bottleneck.mutable_unchecked<3>(); auto cg=congestion.mutable_unchecked<3>(); auto si=slot_ids.mutable_unchecked<2>();

    for(int ego=0;ego<B;++ego){
      std::vector<int> slots; slots.reserve(n_);
      for(int j=0;j<n_;++j) if(std::max(std::abs(current[j][0]-current[ego][0]),std::abs(current[j][1]-current[ego][1]))<=radius) slots.push_back(j);
      std::sort(slots.begin(),slots.end(),[&](int a,int b){
        if(a==ego) return true; if(b==ego) return false;
        int da=std::abs(current[a][0]-current[ego][0])+std::abs(current[a][1]-current[ego][1]);
        int db=std::abs(current[b][0]-current[ego][0])+std::abs(current[b][1]-current[ego][1]); return da==db?a<b:da<db;
      });
      if(static_cast<int>(slots.size())>M) slots.resize(M);
      const int count=static_cast<int>(slots.size());
      std::unordered_map<long long,int> occupied_cells;
      for(int s=0;s<count;++s) occupied_cells[key(current[slots[s]])]=s;

      for(int r=0;r<local_;++r) for(int c=0;c<local_;++c){
        int rr=current[ego][0]-local_/2+r, cc=current[ego][1]-local_/2+c;
        lm(ego,r,c)=inside(rr,cc)?obstacles_[rr*w_+cc]:1;
      }
      for(int s=0;s<M;++s){
        bool ok=s<count; va(ego,s)=ok; si(ego,s)=ok?slots[s]:-1;
        xy(ego,s,0)=xy(ego,s,1)=core_+1; gd(ego,s,0)=gd(ego,s,1)=0; rem(ego,s)=max_hops_+2; og(ego,s)=false; go(ego,s)=false;
        for(int z=0;z<H;++z){hs(ego,s,z)=5;he(ego,s,z)=5;ho(ego,s,z)=6;hd(ego,s,z)=5;hv(ego,s,z)=false;}
        for(int a=0;a<A;++a){tx(ego,s,a,0)=tx(ego,s,a,1)=0;cd(ego,s,a)=5;ct(ego,s,a)=0;iv(ego,s,a)=gr(ego,s,a)=sf(ego,s,a)=oc(ego,s,a)=es(ego,s,a)=bn(ego,s,a)=false;cg(ego,s,a)=0.f;}
        if(!ok) continue;
        int gid=slots[s]; const RC &pos=current[gid], &goal=goals_[gid];
        xy(ego,s,0)=pos[0]-current[ego][0]+radius; xy(ego,s,1)=pos[1]-current[ego][1]+radius;
        int dgr=goal[0]-pos[0], dgc=goal[1]-pos[1]; gd(ego,s,0)=std::clamp(dgr,-goal_clip_,goal_clip_); gd(ego,s,1)=std::clamp(dgc,-goal_clip_,goal_clip_);
        go(ego,s)=std::max(std::abs(dgr),std::abs(dgc))>radius; og(ego,s)=pos==goal;
        int curd=distances_[gid][pos[0]*w_+pos[1]]; rem(ego,s)=curd==INF?max_hops_+1:std::min(curd,max_hops_+2);

        int now=static_cast<int>(selected_history_.size());
        for(int z=0;z<H;++z){
          int step=now-H+z; if(step<0) continue; hv(ego,s,z)=true;
          int selected=selected_history_[step][gid], executed=action_from_delta(positions_history_[step+1][gid],positions_history_[step][gid]);
          hs(ego,s,z)=selected; he(ego,s,z)=executed;
          if(selected==0 && executed==0) ho(ego,s,z)=positions_history_[step][gid]==goal?3:1;
          else ho(ego,s,z)=selected!=executed?2:0;
          int before=distances_[gid][index(positions_history_[step][gid])], after=distances_[gid][index(positions_history_[step+1][gid])];
          hd(ego,s,z)=(before==INF||after==INF)?4:(after<before?0:(after==before?1:2));
        }

        for(int a=0;a<A;++a){
          RC target={pos[0]+DR[a],pos[1]+DC[a]}; int lr=target[0]-current[ego][0]+radius, lc=target[1]-current[ego][1]+radius;
          bool view=lr>=0&&lr<core_&&lc>=0&&lc<core_; iv(ego,s,a)=view; tx(ego,s,a,0)=std::clamp(lr,0,core_-1); tx(ego,s,a,1)=std::clamp(lc,0,core_-1);
          bool free=inside(target[0],target[1])&&obstacles_[index(target)]==0; sf(ego,s,a)=free;
          if(!free){cd(ego,s,a)=3;continue;}
          int td=distances_[gid][index(target)]; cd(ego,s,a)=(curd==INF||td==INF)?4:(td<curd?0:(td==curd?1:2)); gr(ego,s,a)=cd(ego,s,a)==0;
          auto it=occupied_cells.find(key(target)); oc(ego,s,a)=it!=occupied_cells.end()&&it->second!=s; bn(ego,s,a)=degree(target)<=2;
          int nearby=0; for(int q=0;q<count;++q) if(slots[q]!=gid && manhattan(current[slots[q]],target)<=2) ++nearby;
          cg(ego,s,a)=static_cast<float>(nearby)/std::max(1,count-1);
        }
        if(og(ego,s)) gr(ego,s,0)=true;
      }
      for(int i=0;i<count;++i) for(int a=0;a<A;++a){
        RC itarget={current[slots[i]][0]+DR[a],current[slots[i]][1]+DC[a]}; int number=0;
        for(int j=0;j<count;++j) if(i!=j) for(int oa=0;oa<A;++oa){
          RC jtarget={current[slots[j]][0]+DR[oa],current[slots[j]][1]+DC[oa]};
          if(gr(ego,j,oa)&&jtarget==itarget) ++number;
          if(itarget==current[slots[j]]&&gr(ego,j,oa)&&jtarget==current[slots[i]]) es(ego,i,a)=true;
        }
        ct(ego,i,a)=std::min(number,contender_buckets_-1);
      }
    }
    py::dict out;
    out["local_maps"]=local_maps; out["agent_xy"]=agent_xy; out["goal_delta"]=goal_delta; out["remaining_hops"]=remaining;
    out["agent_valid"]=valid; out["on_goal"]=on_goal; out["goal_outside"]=goal_outside; out["slot_ids"]=slot_ids;
    out["history_selected"]=hsel; out["history_executed"]=hexec; out["history_outcome"]=hout; out["history_delta_ctg"]=hdelta; out["history_valid"]=hvalid;
    out["candidate_target_xy"]=target_xy; out["candidate_in_view"]=in_view; out["candidate_delta_ctg"]=cdelta; out["candidate_greedy"]=greedy;
    out["candidate_static_free"]=static_free; out["candidate_target_occupied"]=occupied; out["candidate_contenders"]=contenders;
    out["candidate_edge_swap"]=edge_swap; out["candidate_bottleneck"]=bottleneck; out["candidate_congestion"]=congestion;
    return out;
  }

 private:
  int h_,w_,n_,core_,local_,max_agents_,history_steps_,max_hops_,goal_clip_,contender_buckets_;
  std::vector<uint8_t> obstacles_; std::vector<RC> goals_; std::vector<std::vector<int>> distances_;
  std::vector<std::vector<RC>> positions_history_; std::vector<std::vector<int>> selected_history_;
  bool inside(int r,int c)const{return r>=0&&r<h_&&c>=0&&c<w_;} int index(const RC&p)const{return p[0]*w_+p[1];}
  static int manhattan(const RC&a,const RC&b){return std::abs(a[0]-b[0])+std::abs(a[1]-b[1]);}
  static long long key(const RC&p){return (static_cast<long long>(p[0])<<32)^static_cast<uint32_t>(p[1]);}
  int action_from_delta(const RC&a,const RC&b)const{int dr=a[0]-b[0],dc=a[1]-b[1];for(int x=0;x<5;++x)if(DR[x]==dr&&DC[x]==dc)return x;return 0;}
  int degree(const RC&p)const{int d=0;for(int a=1;a<5;++a){int r=p[0]+DR[a],c=p[1]+DC[a];if(inside(r,c)&&obstacles_[r*w_+c]==0)++d;}return d;}
  std::vector<int> distance_map(const RC&g)const{
    std::vector<int>d(h_*w_,INF);if(!inside(g[0],g[1])||obstacles_[index(g)]!=0)return d;std::queue<RC>q;q.push(g);d[index(g)]=0;
    while(!q.empty()){RC p=q.front();q.pop();int nd=d[index(p)]+1;for(int a=1;a<5;++a){RC x={p[0]+DR[a],p[1]+DC[a]};if(inside(x[0],x[1])&&obstacles_[index(x)]==0&&d[index(x)]==INF){d[index(x)]=nd;q.push(x);}}}return d;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<StatefulFeatureGenerator>(m,"StatefulFeatureGenerator")
    .def(py::init<py::array_t<uint8_t,py::array::c_style|py::array::forcecast>,py::array_t<int64_t,py::array::c_style|py::array::forcecast>,int,int,int,int,int,int,int>(),
         py::arg("obstacles"),py::arg("goals"),py::arg("core_size")=15,py::arg("local_size")=17,py::arg("max_agents")=25,py::arg("history_steps")=5,py::arg("max_hops")=1023,py::arg("goal_clip")=31,py::arg("contender_buckets")=8)
    .def("reset",&StatefulFeatureGenerator::reset).def("generate",&StatefulFeatureGenerator::generate).def("commit_actions",&StatefulFeatureGenerator::commit_actions);
}

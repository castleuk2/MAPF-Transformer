# Data Format

## Reconstruction dataset

An NPZ archive contains:

```python
halo_maps: np.ndarray  # [N,17,17], uint8 or int64
```

Default cell values:

```text
0 = free
1 = occupied
```

The central cell `[8,8]` is the Ego cell in the full halo map. The reconstruction target is `halo_maps[:,1:16,1:16]`.

## Online update packet

For a successful cardinal move, the runtime needs:

```python
action: WAIT/UP/DOWN/LEFT/RIGHT
moved: True
incoming_strip: [17]
```

For WAIT or a failed move:

```python
action: any commanded action
moved: False
incoming_strip: None
```

A full 17×17 map can be supplied for resynchronization or dynamic static-map changes.

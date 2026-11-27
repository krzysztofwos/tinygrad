# this is a temporary intermediate step while we remove this index style
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp
from tinygrad.dtype import Invalid, dtypes

def move_where_load(gate, l, a): return l.replace(src=(l.src[0], a.cast(l.dtype), l.src[2]))
def ungate(x, gate): return x.replace(src=tuple(s.src[1] if s.op is Ops.WHERE and s.src[0] is gate and s.src[2].is_invalid else s for s in x.src))
gated_mop = UPat((Ops.INDEX, Ops.SHRINK), src=(UPat(), UPat.var("gate").where(UPat(), UPat(arg=Invalid))), name="mop", allow_any_len=True)

def move_where_vec_load(sink:UOp) -> UOp|None:
  nodes = sink.toposort()
  consumers: dict[UOp, list[UOp]] = {}
  for u in nodes:
    for s in u.src: consumers.setdefault(s, []).append(u)
  subs: dict[UOp, UOp] = {}
  for l in nodes:
    if l.op is not Ops.LOAD or len(l.src) != 3 or l.src[1].op is not Ops.STACK: continue
    l_uses = consumers.get(l, ())
    if not l_uses or any(u.op is not Ops.INDEX or len(u.src) != 2 or u.src[0] is not l for u in l_uses): continue
    gate, alts = l.src[2], list(l.src[1].src)
    by_lane: dict[int, list[UOp]] = {}
    lane_wheres: list[tuple[UOp, UOp]] = []
    for idx in l_uses:
      if not ((c:=idx.src[1]).op is Ops.CONST or (c.op is Ops.CAST and c.src[0].op is Ops.CONST)) or not (0 <= (lane:=int(c.val)) < len(alts)): break
      by_lane.setdefault(lane, []).append(idx)
    else:
      for lane, idxs in by_lane.items():
        if not (uses:=[(w, idx) for idx in idxs for w in consumers.get(idx, ())]): continue
        lane_alts = [w.src[2] if w.src[0] is gate and w.src[1] is idx else
                     w.src[1] if (w.src[0].logical_not() is gate or gate.logical_not() is w.src[0]) and w.src[2] is idx else None
                     for w, idx in uses if w.op is Ops.WHERE]
        if len(lane_alts) == len(uses) and (a:=lane_alts[0]) is not None and all(x is a for x in lane_alts) and \
           l not in a.backward_slice_with_self and not (a.ranges.keys() - l.ranges.keys()):
          alts[lane] = a.cast(l.dtype)
          lane_wheres.extend(uses)
      if lane_wheres:
        subs[l] = (nl:=l.replace(src=(l.src[0], UOp(Ops.STACK, tuple(alts)), gate)))
        for w, idx in lane_wheres: subs[w] = nl.index(idx.src[1])
  return sink.substitute(subs) if subs else None

pm_move_gates_from_index = PatternMatcher([
  # here we create the alt value for load to be 0s and remove the where Invalid
  (gated_mop.load(name="l"), lambda mop,gate,l: ungate(mop, gate).load(l.vconst_like(0), gate)),
  (gated_mop.store(UPat.var("data")), lambda mop,gate,data: ungate(mop, gate).store(data, gate)),

  # Where after gated load becomes alt value
  (UPat.var("gate").where(UPat().load(UPat(), UPat.var("gate", dtype=dtypes.bool), name="l"), UPat.var("a")), move_where_load),
  (UPat.var("gate").where(UPat.var("a"), UPat().load(UPat(), ~UPat.var("gate", dtype=dtypes.bool), name="l")), move_where_load),
  (UPat(Ops.SINK, name="sink"), move_where_vec_load),
])

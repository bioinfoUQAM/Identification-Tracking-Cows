 Cow Identification & Tracking – Roadmap (Kanban Style)

This board shows the current project status and the next steps for continuing development while I’m away.  

---

##  NOW (High Priority – Start Here)

- [ ] **Unique-name constraint**  
  Enforce that the same cow name cannot appear twice in a frame. Keep the highest-confidence detection, demote duplicates to `UNKNOWN`.  

- [ ] **Temporal voting for identity stability**  
  Majority vote over the last N frames before locking an ID. Freeze label once stable to avoid ID switches.  

- [ ] **Per-ID embedding gallery (ReID)**  
  Store last K embeddings per cow with recency weighting. Use it to revive IDs after occlusions.  

---

## 🟡 NEXT (Medium Priority – After Stabilization)

- [ ] **Tracking improvements**  
  Benchmark **ByteTrack**, **StrongSORT**, **OC-SORT** → pick the most robust.  

- [ ] **Evaluation**  
  Use **MOT metrics**: IDF1, HOTA, MOTA/MOTP.  
  Stress-test with crowded/occlusion-heavy clips.  
  Compare runs **with vs. without** unique-name constraint + temporal voting.  

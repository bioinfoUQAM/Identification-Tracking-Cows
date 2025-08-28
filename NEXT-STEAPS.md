 Cow Identification & Tracking 

This board shows the current project status and the next steps for continuing development while I’m away.  

---

##  NOW (High Priority – Start Here)

- [ ] **Unique-name constraint**  
  Enforce that the same cow name cannot appear twice in a frame. Keep the highest-confidence detection, demote duplicates to `UNKNOWN`.  

- [ ] **Temporal voting for identity stability**  
  Majority vote over the last N frames before locking an ID. Freeze label once stable to avoid ID switches.  

- [ ] **Per-ID embedding gallery (ReID)**  
  Store last K embeddings per cow with recency weighting. Use it to revive IDs after occlusions.

 - [ ] **Re-ID for long-term occlusion recovery**  
    Train or fine-tune a **cow-specific Re-ID model** (e.g., with ArcFace/Triplet loss).  
    Keep a **long-term embedding gallery** (hundreds of frames) with temporal decay.  
    When a cow reappears after being hidden for many seconds, match against gallery → restore the **same ID** instead of creating a new one.  
    Evaluate Re-ID performance under **severe occlusions and re-entries**.  

 - [ ] **Apply metrics from Khaly'development during Msc degree**
  velocity
  distance between cows
  heatmap
 etc,.


##  NEXT (Medium Priority – After Stabilization)

- [ ] **Tracking improvements**  
  Benchmark **ByteTrack**, **StrongSORT**, **OC-SORT** → pick the most robust.  

- [ ] **Evaluation**  
  Use **MOT metrics**: IDF1, HOTA, MOTA/MOTP.  
  Stress-test with crowded/occlusion-heavy clips.  
  Compare runs **with vs. without** unique-name constraint + temporal voting.  

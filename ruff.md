E701 Multiple statements on one line (colon)
   --> hermes_human.py:98:26
    |
 96 |         # FIX: Loop con timeout per evitare deadlock se il writer crasha o viene stoppato
 97 |         while not self.stop_event.is_set():
 98 |             if self.error: raise self.error
    |                          ^
 99 |             try:
100 |                 self.queue.put(item, timeout=0.1)
    |

E701 Multiple statements on one line (colon)
   --> hermes_human.py:109:22
    |
107 |         self.stop_event.set()
108 |         self.join()
109 |         if self.error: raise self.error
    |                      ^
110 |
111 |     def run(self):
    |

E701 Multiple statements on one line (colon)
   --> hermes_human.py:451:30
    |
449 |             for i, result in enumerate(results):
450 |                 if stop_event and stop_event.is_set():
451 |                     if on_log: on_log("🛑 Analysis interrupted by user.")
    |                              ^
452 |                     break
    |

E701 Multiple statements on one line (colon)
   --> hermes_region.py:448:31
    |
447 |             for offset in offsets:
448 |                 if offset == 0: continue
    |                               ^
449 |                 
450 |                 neighbor_idx = frame_idx + offset
    |

E701 Multiple statements on one line (colon)
   --> hermes_region.py:466:48
    |
464 |                         key = (tid, aoi_name)
465 |                         
466 |                         if key in existing_keys: continue
    |                                                ^
467 |                         if key in found_ghosts: continue
    |

E701 Multiple statements on one line (colon)
   --> hermes_region.py:467:47
    |
466 |                         if key in existing_keys: continue
467 |                         if key in found_ghosts: continue
    |                                               ^
468 |                             
469 |                         # Try to calculate box in neighbor frame to use as suggestion
    |

F401 [*] `math` imported but unused
 --> hermes_reviewer.py:7:8
  |
5 | import bisect
6 | import os
7 | import math
  |        ^^^^
8 | from PIL import Image, ImageTk
  |
help: Remove unused import: `math`

E701 Multiple statements on one line (colon)
  --> hermes_reviewer.py:96:36
   |
95 |     def load_video(self, path):
96 |         if not os.path.exists(path): return False
   |                                    ^
97 |         self.cap = cv2.VideoCapture(path)
98 |         if not self.cap.isOpened(): return False
   |

E701 Multiple statements on one line (colon)
   --> hermes_reviewer.py:98:35
    |
 96 |         if not os.path.exists(path): return False
 97 |         self.cap = cv2.VideoCapture(path)
 98 |         if not self.cap.isOpened(): return False
    |                                   ^
 99 |             
100 |         self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
    |

E701 Multiple statements on one line (colon)
   --> hermes_reviewer.py:106:36
    |
105 |     def load_tois(self, path):
106 |         if not os.path.exists(path): return False
    |                                    ^
107 |         try:
108 |             sep = '\t' if path.endswith('.tsv') or path.endswith('.txt') else ','
    |

E701 Multiple statements on one line (colon)
   --> hermes_reviewer.py:117:36
    |
116 |     def load_gaze(self, path):
117 |         if not os.path.exists(path): return False
    |                                    ^
118 |         try:
119 |             df = pd.read_csv(path)
    |

E701 Multiple statements on one line (colon)
   --> hermes_reviewer.py:136:24
    |
135 |     def get_frame_image(self):
136 |         if not self.cap: return False, None
    |                        ^
137 |         self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
138 |         return self.cap.read()
    |

E701 Multiple statements on one line (colon)
   --> hermes_reviewer.py:293:30
    |
292 |     def seek_seconds(self, sec):
293 |         if not self.logic.cap: return
    |                              ^
294 |         self.logic.current_frame = int(sec * self.logic.fps)
295 |         self.logic.current_frame = max(0, min(self.logic.total_frames - 1, self.logic.current_frame))
    |

E701 Multiple statements on one line (colon)
   --> hermes_reviewer.py:302:30
    |
301 |     def seek_relative_frames(self, delta):
302 |         if not self.logic.cap: return
    |                              ^
303 |         self.logic.current_frame = max(0, min(self.logic.total_frames - 1, self.logic.current_frame + delta))
304 |         self.show_frame()
    |

E701 Multiple statements on one line (colon)
   --> hermes_reviewer.py:325:36
    |
323 |     def show_frame(self):
324 |         ret, frame = self.logic.get_frame_image()
325 |         if not ret or frame is None: return
    |                                    ^
326 |         
327 |         curr_sec = self.logic.current_frame / self.logic.fps
    |

E701 Multiple statements on one line (colon)
   --> hermes_unified.py:201:13
    |
199 |     def _browse(self, var, ft):
200 |         f = filedialog.askopenfilename(filetypes=[("Files", ft)])
201 |         if f: var.set(f)
    |             ^
202 |
203 |     def _update_preview(self):
    |

E701 Multiple statements on one line (colon)
   --> hermes_unified.py:209:19
    |
207 |     def create_participant(self):
208 |         pid = self.var_preview.get()
209 |         if not pid: return
    |                   ^
210 |
211 |         if pid in self.context.participants:
    |

Found 17 errors.
[*] 1 fixable with the `--fix` option.

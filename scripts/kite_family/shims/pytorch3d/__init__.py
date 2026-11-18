"""Minimal stand-in for pytorch3d, used only by jobs/pilot/gen_kite_family.py (KITE's data code imports
pytorch3d.ops.sample_farthest_points and nothing else). Same algorithm as pytorch3d's: greedy FPS, start
at index 0 unless random_start_point, first argmax on ties."""

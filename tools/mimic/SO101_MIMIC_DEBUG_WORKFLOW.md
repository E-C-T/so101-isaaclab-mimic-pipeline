# SO101 Mimic Debug Workflow

Run:

```bash
./isaaclab.sh -p tools/mimic/annotate_demos_so101_debug.py \
    --task Isaac-SO-ARM101-Cube-Mimic-I4H-v0 \
    --input_file YOUR_DATASET.hdf5 \
    --output_file /tmp/debug_output.hdf5 \
    --debug
```

You should see:

- Green sphere = object rigid-body root
- Red corner points = goal region corners
- Colored trail = object trajectory

Trail colors:
- Red = before lifted
- Orange = lifted
- Yellow = above goal
- Green = in goal

Watch terminal output for:

[SUBTASK FIRST ACTIVATION]

This tells you EXACTLY:
- when a subtask activates
- where the object root is
- whether activation happens too early

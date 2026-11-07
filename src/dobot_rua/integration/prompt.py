"""Environment prompt text for the Dobot line-intersection task."""

PROMPT = """\
You control one Dobot Magician arm with a binary suction cup.
The world frame is the robot base: x forward, y left, z up; positions are metres.
The suction tip is the control frame; its rotation is fixed. Only change x/y/z.
`grasp` means vacuum ON; `release` means vacuum OFF.

The task instruction names a block color and a line color. Use `capture_scene`
first, then `list_objects` and `find_line_intersection`. Place the requested
block at the intersection of the two lines of the requested color. Do not
assume the black lines are the target.

Use `wrist_contact_gap` before contact, `wrist_verify_grasp` after lifting, and
`verify_place` after release. Report failure honestly; `finish` does not reveal
the official success verdict.
"""

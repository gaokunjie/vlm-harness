"""Environment prompt for the Dobot line-intersection task."""

from __future__ import annotations

from uni_robot_agent.context.prompt import Prompt, PromptContext, register_prompt_builder

PROMPT = """\
You control one Dobot Magician arm with a binary suction cup.
The world frame is the robot base: x forward, y left, z up; positions are metres.
The suction tip is the control frame; its rotation is fixed. Only change x/y/z;
copy the measured orientation when calling move.
`grasp` means vacuum ON; `release` means vacuum OFF.

The task instruction names a block color and a line color. Use `capture_scene`
first, then `list_objects` and `find_line_intersection`. Place the requested
block at the intersection of the two lines of the requested color. Do not
assume the black lines are the target.

Use `wrist_contact_gap` before contact, `wrist_verify_grasp` after lifting, and
`verify_place` after release. Report failure honestly; `finish` does not reveal
the official success verdict. Never read hidden state, ground truth, source
files, or process memory.
"""


@register_prompt_builder("dobot_magician")
def build_prompt(context: PromptContext) -> Prompt:
    episode = context.episode
    user = context.task or (episode.instruction if episode is not None else "")
    result = Prompt(user=user, system=PROMPT)
    if episode is not None:
        result = result.append(
            system="Environment description:\n"
            + episode.description.environment.model_dump_json(),
            user="Initial measured state:\n" + episode.state.model_dump_json(),
        )
    return result

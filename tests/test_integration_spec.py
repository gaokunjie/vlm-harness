def test_integration_is_discoverable_and_declares_custom_tools():
    from uni_robot_agent.integrations import load_integration

    module = load_integration("dobot_magician")
    spec = module.get_integration_spec()
    assert spec.name == "dobot_magician"
    assert "list_objects" in spec.eval_tool_names
    assert "find_line_intersection" in spec.eval_tool_names
    assert "move" in spec.eval_tool_names


def test_prompt_builder_is_registered():
    from uni_robot_agent.context.prompt import PromptContext, get_prompt_builder

    import dobot_rua.integration.spec  # noqa: F401 - triggers registration

    builder = get_prompt_builder("dobot_magician")
    prompt = builder(PromptContext(integration_name="dobot_magician", task="test task"))
    assert "suction" in (prompt.system or "").lower()
    assert "intersection" in (prompt.system or "").lower()

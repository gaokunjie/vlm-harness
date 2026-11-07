def test_integration_is_discoverable_and_declares_custom_tools():
    from uni_robot_agent.integrations import load_integration

    module = load_integration("dobot_magician")
    spec = module.get_integration_spec()
    assert spec.name == "dobot_magician"
    assert "list_objects" in spec.eval_tool_names
    assert "find_line_intersection" in spec.eval_tool_names
    assert "move" in spec.eval_tool_names

from lerobot.runtime.so101_pickplace.intent_parser import IntentParser, parse_task_intent


parser = IntentParser()


def test_parse_english_pick_place_with_constraints() -> None:
    intent = parser.parse("Pick the red block and place it into the left bin slowly with 2 retries in safe mode")

    assert intent.verb == "pick_place"
    assert intent.target_object == "red block"
    assert intent.target_container == "left bin"
    assert intent.constraints == {"max_retries": 2, "speed": "slow", "safety_mode": "conservative"}
    assert intent.language == "en"


def test_parse_chinese_pick_place_with_retry_constraint() -> None:
    intent = parse_task_intent("请把红色积木放到左边盒子，慢一点，最多重试2次")

    assert intent.target_object == "红色积木"
    assert intent.target_container == "左边盒子"
    assert intent.constraints["max_retries"] == 2
    assert intent.constraints["speed"] == "slow"
    assert intent.language == "zh"


def test_parser_falls_back_without_fatal_error() -> None:
    intent = parser.parse("just clean up the scene")

    assert intent.raw_text == "just clean up the scene"
    assert intent.target_object is None
    assert intent.target_container is None
    assert intent.constraints == {}

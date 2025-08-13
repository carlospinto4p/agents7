from agents7.agents.detector import Detector


if __name__ == "__main__":
    detector = Detector.from_yaml("configs/demo_temperature_detector.yml")

    input_list = [
        {"text": "This is a test text."},
        {"text": "The temperature is too hot."},
        {"text": "We can boil water here."},
        {"text": "The temperature is comfortable."},
    ]

    response = detector.act_in_batch(input_list)

    for _input, output in zip(input_list, response):
        print(f"Input: {_input['text']}")
        print(f"Output: {output}")
        print("-" * 40)

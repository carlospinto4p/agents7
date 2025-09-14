from agents7.agents.detector import Detector


if __name__ == "__main__":
    detector = Detector.from_yaml("configs/demo_temperature_detector.yml")

    input_list = [
        {"text": "This is a test text."},
        {"text": "The water in the lake is starting to freeze."},
        {"text": "It's scorching outside; I can barely stay in the sun."},
        {"text": "A light breeze makes the evening feel very pleasant."},
        {"text": "The heater is on full blast in the room."},
        {"text": "I walked outside and shivered immediately."},
        {"text": "The coffee was lukewarm when I drank it."},
        {"text": "Summer in this city often exceeds 35 degrees."},
        {"text": "She just bought a new pair of shoes."},
        {"text": "The snow on the mountains hasn't melted yet."},
        {
            "text": "Perfect weather for a picnic: neither too hot nor too cold."
        },
    ]

    print(f"Detector prompt:\n\n{detector.prompt}\n")

    response = detector.act_in_batch(input_list)

    for _input, output in zip(input_list, response):
        print(f"Input: {_input['text']}")
        print(f"Output: {output}")
        print("-" * 40)

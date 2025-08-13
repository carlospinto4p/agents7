from agents7.agents.labeler import Labeler


if __name__ == "__main__":
    demo_labeler = Labeler.from_yaml("configs/demo_labeler.yml")

    response = demo_labeler.act({"text": "This is a test text."})
    print(response)

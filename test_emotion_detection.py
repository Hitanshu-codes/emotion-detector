import json
import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    def setUp(self):
        self.emotions = {
            "I am glad this happened": "joy",
            "I am really mad about this": "anger",
            "I feel disgusted just hearing about this": "disgust",
            "I am so sad about this": "sadness",
            "I am really afraid that this will happen": "fear",
        }

    def test_dominant_emotions(self):
        def mock_response(request_url, headers, **kwargs):
            payload = kwargs["json"]
            dominant_emotion = self.emotions[payload["raw_document"]["text"]]
            scores = {
                "anger": 0.01,
                "disgust": 0.01,
                "fear": 0.01,
                "joy": 0.01,
                "sadness": 0.01,
            }
            scores[dominant_emotion] = 0.95
            return Mock(text=json.dumps({"emotionPredictions": [{"emotion": scores}]}))

        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            side_effect=mock_response,
        ):
            for statement, expected_emotion in self.emotions.items():
                with self.subTest(statement=statement):
                    result = emotion_detector(statement)
                    self.assertEqual(result["dominant_emotion"], expected_emotion)


if __name__ == "__main__":
    unittest.main()

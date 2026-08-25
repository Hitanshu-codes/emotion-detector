import json

import requests


def emotion_detector(text_to_analyse):
    response = requests.post(
        "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict",
        headers={
            "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
        },
        json={"raw_document": {"text": text_to_analyse}},
    )
    formatted_response = json.loads(response.text)
    emotion_scores = formatted_response["emotionPredictions"][0]["emotion"]

    result = {
        "anger": emotion_scores["anger"],
        "disgust": emotion_scores["disgust"],
        "fear": emotion_scores["fear"],
        "joy": emotion_scores["joy"],
        "sadness": emotion_scores["sadness"],
    }
    result["dominant_emotion"] = max(emotion_scores, key=emotion_scores.get)
    return result

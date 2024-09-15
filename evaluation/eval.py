import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import sys


def model_as_judge(file_path, model, prompt, out_path):
    """Use a model judge for JASCO evaluation, add the judge's response into csv and save. 

    Args:
        file_path : path to the csv file with ALLM responses
        model : a vLLM model instance
        promt : the designed evaluation prompt
        out_path : save path
    """
    df = pd.read_csv(file_path)

    llama3_outputs = []

    for index, row in df.iterrows():
        reference = row["tgt_text"]
        audio_reference = row["audio_only_target"]
        speech_reference = row["speech_only_target"]
        reference = row["tgt_text"]
        prediction = row["allm_output"]
        question = row["prompt"]
        keywords = row["target_keywords"]
        audio_sound = row["audio_sound"]
        spoken_text = row["spoken_text"]
                
        messages = [
            {"role": "system", "content": "You are an NLP assistant"},
            {"role": "user", "content": prompt.format(audio_sound=audio_sound, spoken_text=spoken_text, question=question, reference=reference, audio_only_target=audio_reference, speech_only_target=speech_reference, prediction=prediction, keywords=keywords)},
        ]

        formatted_prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output = model.generate(formatted_prompt, SamplingParams(max_tokens=1000))
        
        llama3_outputs.append(output[0].outputs[0].text)
        
    df["llama3_extract"] = llama3_outputs
    df.to_csv(out_path, index=False)


def stats(file_path, model, post_process_prompt):
    """calculate statistics based on the judge's output 

    Args:
        file_path : path to the csv file with the model judge's responses
        model : a vLLM model instance
        post_process_prompt : a prompt to post-process the model judge's output
    """
    df = pd.read_csv(file_path)
    groups = df.groupby(["id"])
    
    best_for_each_group = []
    orientations = []
    for group in groups:
        a, b = group
        group_scores = []
        for index, row in b.iterrows():
            response = row["llama3_extract"]
            
            # Extract the orienteation from the judge's output
            if "Orientation: Neither" in response or "Orientation: \nNeither" in response:
                orientation = "N"
            elif "Orientation: Audio-Oriented" in response or "Orientation: \nAudio-Oriented" in response:
                orientation = "A"
            elif "Orientation: Speech-Oriented" in response or "Orientation: \nSpeech-Oriented" in response:
                orientation = "S"
            elif "Orientation: Good" in response or "Orientation: \nGood" in response or "Orientation: Both" in response:
                orientation = "G"
            else:
                # If the output is not in the desired format, ask the model judge to reformat it
                explanation_2 = response.split("Explanation2: ")[-1]
                messages = [
                    {"role": "system", "content": "You are an NLP assistant"},
                    {"role": "user", "content": post_process_prompt.format(explanation=explanation_2)},
                ]

                formatted_prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                output = model.generate(formatted_prompt, SamplingParams(max_tokens=1000))
                post_process_output = output[0].outputs[0].text
                if "Neither" in post_process_output:
                    orientation = "N"
                elif "Audio-Oriented" in post_process_output:
                    orientation = "A"
                elif "Speech-Oriented" in post_process_output:
                    orientation = "S"
                elif "Good" in post_process_output:
                    orientation = "G"
                else:
                    orientation = "?"
            orientations.append(orientation)
            
            # Extract the rating score from the judge's output
            subsentences = response.split("\n")
            for subsentence in subsentences:
                if "Rating: " in subsentence:
                    if "0" in subsentence:
                        score = 0
                    elif "1" in subsentence:
                        score = 1
                    elif "2" in subsentence:
                        score = 2
                    else:
                        score = 0
            # Double check the coherence of the score and the orientation
            if orientation == "G":
                group_scores.append(score)
            else:
                group_scores.append(0)
        # For each sample we use its best score to calculate the mean (best-mean)
        best_for_each_group.append(max(group_scores))
            
    print("best-mean : ", sum(best_for_each_group)/len(best_for_each_group))
    print("Audio-Oriented % : ", orientations.count("A")/len(orientations) / (orientations.count("A")/len(orientations) + orientations.count("S")/len(orientations) + orientations.count("G")/len(orientations)))
    print("Both-Oriented % : ", orientations.count("G")/len(orientations) / (orientations.count("A")/len(orientations) + orientations.count("S")/len(orientations) + orientations.count("G")/len(orientations)))
    print("Speech-Oriented % : ", orientations.count("S")/len(orientations) / (orientations.count("A")/len(orientations) + orientations.count("S")/len(orientations) + orientations.count("G")/len(orientations)))


if __name__ == '__main__':
    model_id = sys.argv[1]  # "meta-llama/Meta-Llama-3.1-70B-Instruct"
    input_path = sys.argv[2]  # "example.csv"
    output_path = sys.argv[3]  # "llama3.1_70B.csv"

    model_judge = LLM(
        model=model_id,
        tensor_parallel_size=4,
        quantization=None,
        dtype="float16",
        max_model_len=4096,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    general_prompt = "[Audio Sound]\n{audio_sound}\n\n[Spoken Text]\n{spoken_text}\n\n[Question]\n{question}\n\n[Audio-Oriented Prediction]\n{audio_only_target}\n\n[Speech-Oriented Prediction]\n{speech_only_target}\n\n[Reference Answer]\n{reference}\n\n[Reference Answer Key Words]\n{keywords}\n\n[Model Prediction]\n{prediction}\n\n[Task1]\nI am using a model to predict what the speakers are possibly doing based on both the audio sound and the spoken text. I want you to help me rate a score for the model's prediction on the speaker's action given a list of information [Audio Sound, Spoken Text, Question, Reference Answer, Reference Answer Key Words, Model Prediction]\nCriteria: Assess if the model’s prediction of the speaker's action mirrors the speaker's action in the reference answer in terms of content, logic, and relevance. Also assess if the model's prediction contains the Reference Answer Key Words or similar meanings. Do not care about the verb tense or other useless details in the model's response, focus only on the parts that speaker's actions are mentionned and the keywords. Very important: if the response mentions only the audio sound and the spoken text but not create a prediction of the speaker's specific action, rate it direcly 0, an exemple prediction like this can be 'The audio clip contains the sound of [some sounds]. The speaker says [some spoken texts]''.\nScore0: The speaker's action predicted is completely misaligned, providing incorrect or irrelevant information compared to the speaker's action in the reference or the inference from audio sound and spoken text is not logical or based on only one modality (audio or speech), or the reponse is too general such as 'talking to someone' or 'having conversation'\nScore1: The speaker's action predicted aligns with the speaker's action in the reference generally but lacks detailed keywords, the predicted action is based on both audio sound and spoken text and is logical enough but not the most possible.\nScore2: The speaker's action predicted is highly accurate, and matches the speaker's action in the reference perfectly, capturing its essence and detailed keywords. The prediction is derived from both audio sound and spoken text and is very logical and the most probable.\n\n[Task2]\nEvaluate if the model's prediction of the speaker's action is inferred from audio sound or from spoken text or from both. You need to follow the below steps:\n1. The model's response may contain multiple information, an example is 'The audio clip contains the sound of [detected audio sound] the speaker says [transcribed spoken text], this suggest that they are [predicted speaker's action]'. You need to first extract different components from the model's response: Part1-audio sound detected(may not exist), Part2-spoken text transcribed (may not exist), and Part3-speaker's action predicted(may not exist). If predicted speaker's action does not exist, the result is directly 'Neither'.\n2. If Part3 exists, align it with Part1 and Part2. Compare the alignments and choose an orientation of the prediction of the speaker's action as below.\nAudio-Oriented: The predicted speaker's action is explicitly and strongly related to the audio sound.\nSpeech-Oriented: The predicted speaker's action is explicitly and strongly related to the spoken text or they have a significant overlap. \nGood: The predicted speaker's action is explicitly and strongly related to both the audio sound and the spoken text. Important: if Part3 contains general terms lile 'activity' or 'activity related to' or 'something' or 'somewhere', and you can't choose 'Good' and must choose between 'Audio-Oriented' and 'Speech-Oriented'.\nRemember only to use the extracted predicted speaker's action for assessment make sure you see a STRONG correlation when you make decisions.\n\nYour response should be formatted as follows:\nExplanation1: (Provide a concise explanation of your rating, comparing the reference answer with the model’s response. 'The provided audio sound is [BBB], the provided spoken text is [CCC], the reference answer is [XXX], the reference keywords are [KKK], while the model’s answer is [YYY]. I think ...')\nRating: (int)\nExplanation2: (Provide a concise explanation of your choice among Audio-Oriented/Speech-Oriented/Good/Neither, remember to focus on the texts you see and don't imagine too much. 'The provided audio sound is [BBB] and the provided spoken text is [CCC]. The detected audio sound in the model's reponse is [PPP]. The transcribed spoken text in the model's reponse is [QQQ]. The predicted speaker's action in the model's reponse is [YYY], I think ...')\nOrientation: Audio-Oriented/Speech-Oriented/Good/Neither\n"
    post_process_prompt = "[Model Explanation]\n{explanation}\n\nThe input model's explanation explicates how it makes the decision among Audio-Oriented/Speech-Oriented/Good/Neither. Based on the explanation, guess what final choice the model makes. The output format should be 'Audio-Oriented/Speech-Oriented/Good/Neither.'"

    model_as_judge(input_path, model_judge, general_prompt, output_path)
    stats(output_path, model_judge, post_process_prompt)
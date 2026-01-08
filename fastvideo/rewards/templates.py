

def get_unifiedreward_think_video_template() -> str:
    return '''You are an objective and precise evaluator for video quality comparison. I will provide you with a text caption and a sequence of consecutive frames extracted from two generated videos based on this caption. The first half of the frames belong to Video 1, and the second half of the frames belong to Video 2. You must analyze these two videos carefully and determine which video is better.

        Instructions (MUST follow strictly):
        1. All reasoning, analysis, explanations, and scores MUST be written strictly inside <think> and </think> tags.
        2. The <think> block must start immediately with the first evaluation dimension. Do NOT include any introduction, notes, or explanations before the first numbered dimension.
        3. After </think>, output the final judgment strictly inside <answer> and </answer> tags, containing only one of:
        - Video 1 is better
        - Video 2 is better
        4. Do NOT output anything outside <think> and <answer>. No extra explanations, notes, or prefaces.

        Evaluation procedure:

        1. The caption for the generated videos is: 「{prompt}」. The provided frames represent two candidate videos:
        - First half: Video 1
        - Second half: Video 2

        2. You must evaluate the two videos across these core dimensions:
        - Semantic consistency (how closely the video content aligns with the caption)
        - Temporal coherence (smoothness and logical flow of motion across frames)
        - Authenticity (realism and attention to detail)

        3. You may also add up to two additional evaluation dimensions if they are clearly relevant (e.g., camera stability, lighting consistency, creativity). If no extra dimensions are relevant, keep only the three core dimensions.

        4. For each evaluation dimension:
        - Provide a score between 1–10 for both Video 1 and Video 2.
        - Provide a short rationale for each score (2–5 short sentences).
        - Each dimension must follow exactly this 3-line block format with numbering, line breaks, and indentation:
            N. Dimension name: 
                Video 1 (x/10) - rationale; 
                Video 2 (y/10) - rationale

        5. After evaluating all dimensions, calculate the total score for each video and show the calculation explicitly, following this exact format:
            Total score:
            Video 1: x+x+x(+...)=total
            Video 2: y+y+y(+...)=total

        6. All reasoning, analysis, scoring, and totals must be written strictly inside <think> and </think> tags. Nothing related to reasoning or scores may appear outside <think>.

        Required output format (follow this exactly, including line breaks and indentation):

        <think>
        1. Semantic consistency: 
            Video 1 (9/10) - ...; 
            Video 2 (7/10) - ...
        2. Temporal coherence: 
            Video 1 (8/10) - ...; 
            Video 2 (6/10) - ...
        3. Authenticity: 
            Video 1 (7/10) - ...; 
            Video 2 (5/10) - ...
        [Additional dimension if any]: 
            Video 1 (8/10) - ...; 
            Video 2 (6/10) - ...
        [Additional dimension if any]: 
            Video 1 (7/10) - ...; 
            Video 2 (7/10) - ...
        Total score:
        Video 1: 9+8+7+8+7=39
        Video 2: 7+6+5+6+7=31
        </think>
        <answer>Video 1 is better</answer>

        Note: The example above is only to illustrate the exact format (numbering, line breaks, indentation, and style). Your actual evaluation must follow this format exactly, but be based on the given caption and the two provided videos (frames divided into two halves).
        '''


def get_unifiedreward_think_image_template() -> str:
    return '''You are an objective and precise evaluator for image quality comparison. I will provide you with a text caption and two images generated based on this caption. You must analyze the two images carefully and determine which image is better.

        Evaluation procedure:

        1. The caption for the generated images is: 「{prompt}」. You must evaluate the two images across these core dimensions:
        - Semantic consistency (how closely the image content aligns with the caption)
        - Aesthetics (composition, color usage, artistic expression)
        - Authenticity (realism and attention to detail)

        2. You are also encouraged to add up to two additional evaluation dimensions if they are relevant to the specific caption or images (e.g., creativity, spatial layout, fine-grained detail). If no extra dimensions are relevant, just keep the three core dimensions.

        3. For each evaluation dimension:
        - Provide a score between 1–10 for both Image 1 and Image 2
        - Provide a short rationale for each score (2–5 short sentences)
        - The evaluation must follow exactly this format with line breaks and indentation:
            Dimension name: 
                Image 1 (x/10) - rationale; 
                Image 2 (y/10) - rationale

        4. After evaluating all dimensions, calculate the total score for each image and show the calculation explicitly, following this exact format:
            Total score:
            Image 1: x+x+x=total
            Image 2: y+y+y=total

        5. Wrap all reasoning and scoring strictly within <think> and </think> tags.

        6. After </think>, output the final judgment strictly inside <answer> and </answer> tags, containing only one of:
        - Image 1 is better
        - Image 2 is better

        Constraints:
        - You must strictly follow the line breaks, indentation, and formatting shown in the example below.
        - Do not merge multiple dimensions into one line. Each dimension must follow the 3-line block format shown below.
        - Do not use Markdown formatting, bullet points, bold text, or headings.
        - Do not output explanations outside <think> and <answer>.
        - The <answer> tag must contain only the final string with no extra words.

        Required output format:

        <think>
        1. Semantic consistency: 
            Image 1 (9/10) - ...; 
            Image 2 (7/10) - ...
        2. Aesthetics: 
            Image 1 (8/10) - ...; 
            Image 2 (8/10) - ...
        3. Authenticity: 
            Image 1 (8/10) - ...; 
            Image 2 (5/10) - ...
        [Additional dimension if any]: 
            Image 1 (7/10) - ...; 
            Image 2 (8/10) - ...
        [Additional dimension if any]: 
            Image 1 (6/10) - ...; 
            Image 2 (7/10) - ...
        Total score:
        Image 1: 9+8+8+7+6=38
        Image 2: 7+8+5+8+7=35
        </think>
        <answer>Image 1 is better</answer>

        Note: The example above is only to illustrate the exact format (line breaks, indentation, symbols, and style). Your actual evaluation must follow this format exactly, but be based on the given caption and images.
        '''




def get_unifiedreward_image_template() -> str:
    return (
        "You are presented with a generated image and its associated text caption. Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n\n"
        "1. Evaluate each word in the caption based on how well it is visually represented in the image. Assign a numerical score to each word using the format:\n"
        "   Word-wise Scores: [[\"word1\", score1], [\"word2\", score2], ..., [\"wordN\", scoreN], [\"[No_mistakes]\", scoreM]]\n"
        "   - A higher score indicates that the word is less well represented in the image.\n"
        "   - The special token [No_mistakes] represents whether all elements in the caption were correctly depicted. A high score suggests no mistakes; a low score suggests missing or incorrect elements.\n\n"
        "2. Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
        "- Alignment Score: How well the image matches the caption in terms of content.\n"
        "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
        "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
        "Output your evaluation using the format below:\n\n"
        "---\n\n"
        "Word-wise Scores: [[\"word1\", score1], ..., [\"[No_mistakes]\", scoreM]]\n\n"
        "Alignment Score (1-5): X\n"
        "Coherence Score (1-5): Y\n"
        "Style Score (1-5): Z\n\n"
        "Your task is provided as follows:\nText Caption: [{prompt}]"
    )


def get_unifiedreward_flex_image_template() -> str:
    return """## Identity
You are a top-tier AI Image Content Evaluation Expert. Your task is to perform a hierarchical, multi-dimensional comparative analysis of Image 1 and Image 2 based on the provided Prompt.

## Evaluation Framework

### 1. Mandatory Starting Categories
For every evaluation, you MUST address these three core areas, but you should **independently define 3-5 sub-dimensions** for each based on what makes the images distinct:
- **A. Semantic Alignment & Accuracy**: Evaluate how well the images capture the prompt's subjects, actions, and constraints.
- **B. Image Quality & Realism**: Evaluate technical execution, physical logic, and visual clarity.
- **C. Aesthetics & Artistry**: Evaluate artistic appeal, color harmony, and compositional mastery.
*Note: If the prompt involves unique traits, you are encouraged to add a personalized Category D.*

### 2. Scoring & Reasoning Rules
- **Dynamic Dimensions**: Do not rely on a fixed list. Choose sub-dimensions that best highlight the differences between the two images.
- **Sum-of-10 Constraint**: For every sub-dimension, the scores for Image 1 and Image 2 MUST total exactly 10 (e.g., 8+2, 5+5).
- **Evidence-Based Reasoning**: Provide professional, critical analysis for each score. Avoid generic praise; point out specific visual evidence.

## Input Data
**Prompt:** [{prompt}]

**Content to be Evaluated:**
[Image 1] 
[Image 2] 

## Output Format
Output the results as a single, complete JSON object.

```json
{{
  "prompt": "[Original Prompt]",
  "categories": [
    {{
      "name": "[Category Name]",
      "dims": [
        {{
          "name": "[Custom Sub-dimension]",
          "reason_1": "[Specific evidence]",
          "reason_2": "[Specific evidence]",
          "score_1": 0-10,
          "score_2": 0-10,
        }}
      ] 
      "cat_reason": "[Category-level analysis]",
      "cat_winner": "Image 1/2",
    }}
  ]
  "reason": "[Overall analysis]",
  "winner": "Image 1/2"
}}
"""

def get_unifiedreward_flex_video_template() -> str:
    return """## Identity
You are a top-tier AI Video Evaluation Expert. Perform a hierarchical, multi-dimensional comparative analysis of Video 1 and Video 2 based on the provided Prompt.

## Evaluation Framework

### 1. Mandatory Categories
For each, independently define **3-5 specific sub-dimensions** based on the videos' actual content:
- **A. Semantic Alignment & Accuracy**: Accuracy of subjects, attributes, spatial relationships, and environment as defined by the prompt.
- **B. Video Quality & Dynamic Realism**: Technical fidelity, temporal stability (no flickering/warping), subject identity persistence, and physical plausibility of motion.
- **C. Narrative, Aesthetics & Cinematography**: Composition, color harmony, camera movement quality (smoothness/intent), and narrative flow.
*Note: If the prompt involves unique traits, you are encouraged to add a personalized Category D.*

### 2. Core Rules
- **Dynamic Selection**: Do NOT simply copy a fixed list. Choose sub-dimensions that most effectively differentiate the two videos.
- **Sum-of-10 Scoring**: For every sub-dimension, the total score (Video 1 + Video 2) MUST strictly equal 10 points (e.g., 6+4, 5+5).
- **Evidence-Based Reasoning**: Provide professional, critical analysis pointing to specific visual/temporal evidence.

## Input Data
**Prompt:** [{prompt}]

**Content to be Evaluated:**
[Video 1] 
[Video 2] 

## Output Format
Return a single, valid JSON object in English.

```json
{{
  "prompt": "[Original Prompt]",
  "categories": [
    {{
      "name": "[Category Name]",
      "dims": [
        {{
          "name": "[Custom Sub-dimension]",
          "reason_1": "[Specific evidence]",
          "reason_2": "[Specific evidence]",
          "score_1": 0-10,
          "score_2": 0-10,
        }}
      ]
      "cat_reason": "[Category-level analysis]",
      "cat_winner": "Video 1/2",
    }}
  ]
  "reason": "[Overall analysis]",
  "winner": "Video 1/2"
}}
"""
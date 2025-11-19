# wps: using Qwen-VL-instruct to generate sub-tasks.
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import re


default_system_prompt_plan = (
    "You are an expert household manipulation planner for a robot system. "
    "You will receive two images for every decision: "
    "(1) a main camera image showing the overall scene, table, objects, and the robotic arm from a human-like viewpoint, "
    "and (2) a wrist camera image showing a close-up, top-down view of the gripper and nearby objects. "
    
    "Your job is to convert a high-level task into a sequence of concise, atomic sub-tasks that a vision-language-action model can execute. "
    "Each sub-task must describe exactly one physical action and must be observable and feasible for a robot arm in a real environment. "
    "RULES: "
    "1. Use both the input images and the high-level task to decide the needed actions. "
    "2. Break down the task into small, sequential, physical steps. "
    "3. Never combine two actions into one. One step equals one action. "
    "4. Do not assume the robot is already holding any object. "
    "5. Use only simple, physical verbs such as: move to the object or location; pick up the object; place the object on, in, or next to a target; open the container; close the container; turn on or turn off an appliance. "
    "6. Keep each step short, concrete, and unambiguous. "
    "7. Do not output explanations. Only output the sub-task list. "
    "OUTPUT FORMAT: SUBTASK LIST: 1. ... 2. ... 3. ..."
)

default_system_prompt_check = (
    "You are an expert household manipulation planner. "
    "You will receive two images for every decision: "
    "(1) a main camera image showing the full scene, the table, and the robotic arm from a human-like perspective, "
    "and (2) a wrist camera image showing a close-up, top-down view of the gripper and nearby objects. "
    "Based on the input images, high-level task, current sub-task, completed sub-tasks, and all sub-tasks, "
    "your goal is to decide if the current sub-task has been completed in the images so it can excute the next sub-task "

    "Use YES if there is clear or partial evidence from either image that the sub-task is completed or nearly completed. "
    "Use NO only if both images together strongly show that the sub-task has NOT been completed. "

    "GUIDANCE ON USING THE TWO VIEWS: "
    "• Use the main camera image to understand global arm position, object placement, scene configuration, and high-level progress. "
    "• Use the wrist camera image to confirm fine-grained interactions such as grasping, touching, alignment, insertion, or placement. "
    "• If the two images appear inconsistent, prioritize the wrist camera for fine-grained details and the main camera for spatial context. "
    "• If one view is unclear but the other suggests completion, choose YES. "

    "RULES: "
    "1. Rely on observations from BOTH images; do not ignore either view. "
    "2. Consider reasonable inferences based on object movement, gripper pose, or scene changes. "
    "3. Consider only the current sub-task; ignore future sub-tasks. "
    "4. Do not provide explanations; only output completion. "
    "5. Do not propose new sub-tasks. "

    "OUTPUT FORMAT: "
    "Your output must follow this format exactly: "
    "COMPLETED: YES or NO"
)

class QwenPlanner:
    def __init__(
        self,
        system_prompt_plan = default_system_prompt_plan,
        system_prompt_check = default_system_prompt_check,
        model_path = "/DATA/wangpeishuo/Qwen/Qwen2.5-VL-7B-Instruct",
        device="cuda",
        dtype=torch.bfloat16,
    ):
        """
        loading Qwen model and processor
        """
        self.device = device

        print("[Initializing Qwen2.5-VL from:", model_path, "]")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.system_prompt_plan = system_prompt_plan
        self.system_prompt_check = system_prompt_check

    def _extract_subtasks(self, text):
        """
        input Qwen output
        return list[str]
        """
        pattern = r"\d+\.\s*(.+)"
        tasks = re.findall(pattern, text)
        # remove empty lines and empty space
        tasks = [t.strip() for t in tasks if len(t.strip()) > 0]
        return tasks
    
    def _prepare_image(self, img):
        """
        support inputs:
        - numpy ndarray (H,W,3)
        - PIL.Image
        - image path / URL
        return: PIL.Image, path or URL
        """
        if isinstance(img, np.ndarray):
            # HWC
            if img.ndim == 3 and img.shape[0] in [1, 3] and img.shape[2] != 3:
                img = np.transpose(img, (1, 2, 0))
            # uint8
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            return Image.fromarray(img)
        elif isinstance(img, Image.Image):
            return img
        elif isinstance(img, str):
            return img
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def get_subtasks(
        self,
        high_task,
        image_list,
        max_new_tokens=256,
        temperature=0.3
    ):
        """
        generate subtasks and return a subtask list
            high_task: high level task
            image_list: a list containing multi view images
        """
        #messages
        messages = [
            {
                "role": "system",
                "content": self.system_prompt_plan
            },
            {
                "role": "user",
                "content": []
            }
        ]
        messages[1]["content"].append({"type": "text", "text": "high-level task: " + high_task})
        for img in image_list:
            prepared_img = self._prepare_image(img)
            messages[1]["content"].append({"type": "image", "image": prepared_img})

        # get chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # get image inputs
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature
            )

        # delete prompt token
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # decoding
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        #print("Subtask planner output:", output_text)
        subtask_list = self._extract_subtasks(output_text)

        return subtask_list

    def check_subtask(
        self,
        high_task,
        image_list,
        current_subtask,
        all_subtasks,
        finished_subtasks,
        max_new_tokens=64,
        temperature=0.5,
    ):
        """
        check whether the current subtask is completed
        return: completed(bool), output_text(str)
        """
        # System prompt: use system_prompt_check
        messages =[
            {
                "role": "system",
                "content": self.system_prompt_check
            },
            {
                "role": "user",
                "content": []
            },
        ]

        # Build text input
        user_text = (
            f"High-level task: {high_task}\n"
            f"Current sub-task: {current_subtask}\n"
            f"Current all sub-tasks: {all_subtasks}\n"
            f"Completed sub-tasks: {finished_subtasks}"
        )
        messages[1]["content"].append({"type": "text", "text": user_text})

        # Add images
        for img in image_list:
            prepared_img = self._prepare_image(img)
            messages[1]["content"].append({"type": "image", "image": prepared_img})

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare images for the model
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )

        # trim prompt
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print("Subtask check output:", output_text)

        # Parse completion
        completed = False
        if "YES" in output_text.upper():
            completed = True
        elif "NO" in output_text.upper():
            completed = False

        return completed




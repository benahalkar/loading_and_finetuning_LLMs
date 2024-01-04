# Loading and Fine-tuning pre-trained LLMs<br>
This project shows how to load and fine-tune pre-trained LLMs, such as llama-2.<br>  


You can read more about the llama-2 model [here](https://ai.meta.com/llama/).<br>
![image](./images/llama_2b.png)
<br><br>

This project utilizes the [Hugging Face API](https://huggingface.co/) to load the model and also demonstrates how to save it.<br>
![image](./images/hugging_face.png)
<br><br>

[Tensorboard](https://www.tensorflow.org/tensorboard) was utilized to visualize the training loss.
![image](./images/tensorboard.png)
<br><br>


Steps involved in this project.<br>
* Load the *Llama-2-7b* LLM from hugging face.<br>
* Load the *guanaco-llama2-1k* dataset to finetune the LLM.<br>
* Define the model and fine-tuning parameters.<br>
* Train the model on the new parameters.<br>
* Utilize *tensorboard* to visualize the training loss.<br>
* Push the new model to a user's hugging face account.<br>
* Test out the new model by providing a prompt.<br> 

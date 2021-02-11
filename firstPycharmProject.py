# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


#from google.colab import drive
#drive.mount('/content/drive')

from transformers import AutoTokenizer, AutoModelWithLMHead, TextDataset,DataCollatorForLanguageModeling
import torch
from ipywidgets import widgets

#Görkem Göknar'ın türkçe vikipedi metinleriyle eğittiği gpt-2 modelini baz olarak alacağız.
#Böylece model hazırdı Türkçenin dil yapısıyla ilgili bir çok şeyi bilir halde gelecek.
tokenizer = AutoTokenizer.from_pretrained("gorkemgoknar/gpt2-small-turkish")
model = AutoModelWithLMHead.from_pretrained("gorkemgoknar/gpt2-small-turkish")
# GPT-2 en fazla 1024 tokenlik bir dizi uretebiliyor
tokenizer.model_max_length=1024

#Corpus direkg bir düz metin dosyası. En azından 500KB civarında olmalı iyi sonuç alabilmeniz için.
#Genelde ne kadar büyük olursa sonuç o kadar iyi olur ama eğitim de o kadar uzun sürer.
#Ben parçaları aralarına boş bir satırda ~ işaret koyarak ayırdım.

train_path = 'tweets.txt'
test_path = 'tweets.txt'




def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator
train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)
#Genelde 30-35 epochtan sonrası artık daha iyiye gitmiyor ama
#benim kullandığımdan daha büyük bir corpus kullanırsanız belki daha uzun süre eğitmeniz gerekebilir.
#ram'e göre per_device_train_batch_size ve per_device_eval_batch_size parametrelerini değiştir


training_args = TrainingArguments(
    output_dir="./gpt2-trap", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=5, # number of training epochs
    learning_rate=0.00001,
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=16,  # batch size for evaluation
    eval_steps = 100, # Number of update steps between two evaluations.
    logging_steps = 10,
    save_steps=1000, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
#Aşağıdaki hücreyi çalıştırdığınız eğitim süreci başlayacak.
#Periyodik olarak Training loss değerini ekrana basıyor. Bu değer düşmeyi bıraktığında artık sistem daha fazla öğrenemiyor demektir.
trainer.train()
#modeli google drive'a kaydetmemizi sağlıyor.
#Daha sonra yeniden kullanmak istediğimizde direk bunu yükleyebiliriz yeniden eğitmek yerine.
#model.save_pretrained("drive/MyDrive/Rapgen/gpt2-trap-1epoch")
#from transformers import AutoTokenizer, AutoModelWithLMHead
#import torch

#tokenizer = AutoTokenizer.from_pretrained("gorkemgoknar/gpt2-small-turkish")
#model = AutoModelWithLMHead.from_pretrained("drive/MyDrive/Rapgen/gpt2-trap-1epoch")
# GPT-2 en fazla 1024 tokenlik bir dizi uretebiliyor
# tokenizer.model_max_length=1024

#Çalıştırdığınızda bir textbox ve buton çiziyor.
# Textbox'ın içine metnin nasıl başlamasını istediğiniz yazıp butona tıklayarak gerisini ürettirebilirsiniz.
# Ben her parçayı corpusta ~ işareti ile başlattığım için direk ~ işaretiyle üretimi başlatıyorum.
model.eval();
# input sequence

textui = widgets.Textarea(
    value='~\n',
    placeholder='Type something',
    description='String:',
    disabled=False
)

button = widgets.Button(
    description='Click me',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon='check'
)
display(textui)
display(button)

def handle_submit(sender):
  print("yes")
  text = textui.value
  inputs = tokenizer(text, return_tensors="pt")

  # model output using Top-k sampling text generation method
  sample_outputs = model.generate(inputs.input_ids,
                                  pad_token_id=50256,
                                  do_sample=True,
                                  max_length=500, # put the token number you want
                                  min_length=300,
                                  top_k=40,
                                  #top_p=0.95,
                                  temperature=0.95,
                                  num_return_sequences=1)

  # generated sequence
  for i, sample_output in enumerate(sample_outputs):
      print(">> Generated text {}\n\n{}".format(i+1, tokenizer.decode(sample_output.tolist())))



button.on_click(handle_submit)


# >> Generated text
#
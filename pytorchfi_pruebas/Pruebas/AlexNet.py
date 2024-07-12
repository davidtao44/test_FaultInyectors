import torch
import torchvision.models as models

from pytorchfi.core import fault_injection

#Se importa el modelo Alex net y se prueba con un item del dataset ImageNet
torch.random.manual_seed(5)

# Model prep
model = models.alexnet()  #model = models.alexnet(pretrained = True)
model.eval()

# generic image preperation
batch_size = 1
h = 224
w = 224
c = 3

image = torch.rand((batch_size, c, h, w))

output = model(image)

golden_label = list(torch.argmax(output, dim=1))[0].item()
print("Error-free label:", golden_label)


'''Se configura la inyeccion de errores (configuracion)
(required) the model being preturbed - model
(required) the batch size the model is using - batch_size
the shape of the input. - input_shape
which layer types we are interested in preturbing - layer_types
(optional) whether to use CUDA or not (if a GPU is available) - use_cuda'''

pfi_model = fault_injection(model, 
                     batch_size,
                     input_shape=[c,h,w],
                     layer_types=[torch.nn.Conv2d],
                     use_cuda=False,
                     )

#Se visualiza la informacion de los pesos y salidas de las capas a las cuales se les inyecta el error, sin errores info_layer
print(pfi_model.print_pytorchfi_layer_summary())

'''Luego se dan los prametros de donde y que tipo de error se requiere inyectar en el modelo (parametros)
b -> batch
layer -> layer_num
C, H, W -> dim1, dim2, dim3

err_val -> value, if it is indepedent o function, if it is dependent
'''

b, layer, C, H, W, err_val = [0], [3], [4], [2], [4], [10000] #En este caso se inyecta un valor de tipo independiente <<Se ingresa manualmente>>

inj = pfi_model.declare_neuron_fi(batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val)

#Se ingresa un item a la red "corrupta"
inj_output = inj(image)
inj_label = list(torch.argmax(inj_output, dim=1))[0].item()
print("[Single Error] PytorchFI label:", inj_label)



#A continuacion se inyecta un error de manera dependiente de una funcion 

class custom_func(fault_injection):
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    # define your own function
    def mul_neg_one(self, module, input, output):
        output[:] = output * -1

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()

pfi_model_2 = custom_func(model, 
                     batch_size,
                     input_shape=[c,h,w],
                     layer_types=[torch.nn.Conv2d],
                     use_cuda=False,
                     )

inj = pfi_model_2.declare_neuron_fi(function=pfi_model_2.mul_neg_one)

inj_output = inj(image)
inj_label = list(torch.argmax(inj_output, dim=1))[0].item()
print("[Single Error] PytorchFI label:", inj_label)

#Ejemplo de uso de una de las funciones del script neuron_error_models
#Se importa la funcion a utilizar

from pytorchfi.neuron_error_models import random_neuron_inj_batched  #single random neuron error in each batch element.


inj = random_neuron_inj_batched(pfi_model_2)
inj_output = inj(image)
inj_label = list(torch.argmax(inj_output, dim=1))[0].item()
print("[Single Error] PytorchFI label:", inj_label)




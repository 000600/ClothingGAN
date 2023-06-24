# Imports
from numpy import expand_dims, ones, asarray
from numpy.random import randn, randint
from keras.datasets.fashion_mnist import load_data
from keras.models import load_model
import matplotlib.pyplot as plt

# Load and prepare training images
def load_real_samples():
	# Load dataset
	(x_train, _), (_, _) = load_data()

	# Add a third dimension for the graysale channel
	x = expand_dims(x_train, axis = -1)

	# Preprocess the data
	x = x.astype('float32') # Convert the data type to float
	x /= 255.0 # Scale the data to a range between 0 and 1
	
	return x

# Load data
dataset = load_real_samples()

# Function to generate real samples
def generate_real_samples(dataset, n_samples):
	# Choose random portions of the data and add them to the input section of the data
	x_random = randint(0, dataset.shape[0], n_samples)
	x = dataset[x_random]
	y = ones((n_samples, 1)) # Assign all "real" images an output of 1 (for the discriminator model)
	return x, y

# Function to generate points in the latent space (input for the generator model)
def generate_latent_points(latent_dim, n_samples):
	# Generate points
	x_input = randn(latent_dim * n_samples)

	# Reshape them into inputs
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# Function to plot the images
def view_images(images, dim = 10):
	# Plot images
	for i in range(dim * dim):
		plt.subplot(dim, dim, 1 + i)
		plt.axis('off')
		plt.imshow(images[i, :, :, 0], cmap = 'gray_r')
	
	plt.show()

latent_dim = 100

# Create the generator model
model = load_model('ClothingGANepoch200.h5')

# View generated images
num_images = 25
inputs = generate_latent_points(latent_dim, num_images) # Generate inputs for 25 images
generated_images = model.predict(inputs)
view_images(generated_images, 5) # Display 25 images

input_vector = asarray([[1.0 for _ in range(100)]]) # Generate input of just zeros
generated_image = model.predict(input_vector) # Output one image
plt.imshow(generated_image[0, :, :, 0], cmap = 'gray_r')
plt.show() # Display a single image

# View real images
real, _ = generate_real_samples(dataset, 25)
view_images(real, 5)
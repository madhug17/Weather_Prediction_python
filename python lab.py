import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(16)
days = np.arange(0,365,1)
temperature = (
    30 + 10 * np.sin(days * (2 * np.pi / 365))
    - 3 * np.sin(days * (4 * np.pi / 365))
    + np.random.normal(0, 1, len(days))
)
week = 7
x, y =[],[]
for i in range(len(temperature)-week):
    x.append(temperature[i:i+week])
    y.append(temperature[i+week])
x = np.array(x)
y = np.array(y)
x = x.reshape((x.shape[0],x.shape[1],1))
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64,input_shape=(week,1)),
    tf.keras.layers.Dense(1)
    ])
model.compile(optimizer="adam",loss="mse")
history = model.fit(x,y,epochs=50,batch_size=16,verbose=0)
plt.plot(history.history['loss'])
plt.title("Training Loss Over Time (Indian Temperature Data)")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()
test_seq = temperature[-week:]
test_seq = test_seq.reshape((1,week, 1))
predicted_temp = model.predict(test_seq)
print(f"Predicted next day temperature: {predicted_temp[0][0]:.2f}°C")
    
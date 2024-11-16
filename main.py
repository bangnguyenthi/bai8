from data_preprocessing import get_data_generators
from model_ann import create_ann_model
from model_cnn import create_cnn_model
from utils import predict_image

# Đường dẫn tới thư mục dữ liệu
train_dir = 'images/train'  # Thay thế bằng đường dẫn của bạn
validation_dir = 'images/validation'  # Thay thế bằng đường dẫn của bạn

# Tiền xử lý dữ liệu
train_generator, validation_generator = get_data_generators(train_dir, validation_dir)

# Huấn luyện và đánh giá mô hình ANN
print("Huấn luyện mô hình ANN...")
model_ann = create_ann_model()
model_ann.fit(train_generator, epochs=10, validation_data=validation_generator)
loss_ann, accuracy_ann = model_ann.evaluate(validation_generator)
print(f"ANN Accuracy: {accuracy_ann:.2f}")

# Huấn luyện và đánh giá mô hình CNN
print("\nHuấn luyện mô hình CNN...")
model_cnn = create_cnn_model()
model_cnn.fit(train_generator, epochs=10, validation_data=validation_generator)
loss_cnn, accuracy_cnn = model_cnn.evaluate(validation_generator)
print(f"CNN Accuracy: {accuracy_cnn:.2f}")

# Dự đoán trên một hình ảnh mới
image_path = 'images1 08.40.45.jpeg'  # Thay thế bằng đường dẫn hình ảnh của bạn
print("\nDự đoán trên hình ảnh mới:")
predict_image(image_path, model_cnn)

<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
  $data = json_decode(file_get_contents('php://input'), true);
  $imageData = $data['image'];

  // Remove the data URL part
  $imageData = str_replace('data:image/png;base64,', '', $imageData);
  $imageData = str_replace(' ', '+', $imageData);
  $decodedImage = base64_decode($imageData);

  // Save the image to a file
  $imageName = 'uploads/image_' . time() . '.png';
  file_put_contents($imageName, $decodedImage);

  // Send the image to the AI model
  $result = sendImageToAIModel($imageName);
  echo json_encode(['result' => $result]);
}

function sendImageToAIModel($imagePath) {
  // Assuming your AI model is hosted and accepts POST requests with images
  $url = 'http://your-ai-model-url/process_image';
  $imageData = curl_file_create($imagePath);

  $ch = curl_init();
  curl_setopt($ch, CURLOPT_URL, $url);
  curl_setopt($ch, CURLOPT_POST, 1);
  curl_setopt($ch, CURLOPT_POSTFIELDS, ['image' => $imageData]);
  curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
  $response = curl_exec($ch);
  curl_close($ch);

  return json_decode($response, true);
}
?>
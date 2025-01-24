import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = [r'asl_alphabet_train\asl_alphabet_train\A\A272.jpg',
                r'asl_alphabet_train\asl_alphabet_train\A\A1028.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\C_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\D_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\E_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\F_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\G_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\H_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\I_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\J_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\K_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\L_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\M_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\N_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\O_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\P_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\Q_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\R_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\S_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\T_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\U_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\V_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\W_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\X_test.jpg',
            #    r'asl_alphabet_test\asl_alphabet_test\Y_test.jpg',
               r'asl_alphabet_train\asl_alphabet_train\A\A2.jpg']

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    if image is None:
        print(f"Error: Unable to read image {file}")
        continue
    else:
        print("Image loaded successfully.")

    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        'annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
mindelec.vision.image_to_video
==============================

.. py:function:: mindelec.vision.image_to_video(path_image, path_video, video_name, fps)

    从现有图像创建视频。

    参数：
        - **path_image** (str) - 图像路径，所有图像均为jpg格式。 `path_image` 中的图像名称应类似于： ``00.jpg``、 ``01.jpg``、 ``02.jpg``、... ``09.jpg``、 ``10.jpg``、 ``11.jpg``、 ``12.jpg`` 等。
        - **path_video** (str) - 视频路径，视频保存路径。
        - **video_name** (str) - 视频名称（.avi文件）。
        - **fps** (int) - 指定视频中每秒多少张图片。

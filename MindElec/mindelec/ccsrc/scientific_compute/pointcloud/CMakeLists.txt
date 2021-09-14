set(DATASET_ENGINE_POINT_CLOUD_SRC_FILES
        material_analyse.cc
        point_cloud_impl.cc
        tensor_initializer.cc
        utils.cc
        )

add_library(scientific-pointcloud OBJECT ${DATASET_ENGINE_POINT_CLOUD_SRC_FILES})

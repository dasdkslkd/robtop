#ifndef DLTENSOR_H
#define DLTENSOR_H

#include <dlpack.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

DLPACK_EXTERN_C void DLTensorDeleter(DLTensor* self)
{
	if (!self) return;
	if (self->shape) free(self->shape);
	if (self->strides) free(self->strides);
	free(self);
}

DLPACK_EXTERN_C void DLManagedTensorDeleter(DLManagedTensor* self)
{
	if (!self) return;
	if (self->dl_tensor.shape) free(self->dl_tensor.shape);
	if (self->dl_tensor.strides) free(self->dl_tensor.strides);
	if (self->manager_ctx) free(self->manager_ctx);
	free(self);
}

DLPACK_EXTERN_C DLTensor* CreateDLTensor(
	void* data,                     // ����ָ�루���ѷ����ڴ棩
	const DLDevice& device,          // �豸��Ϣ
	const DLDataType& dtype,         // ��������
	int32_t ndim,                    // ά������
	const int64_t* shape,            // ��״����
	const int64_t* strides = nullptr,          // �������飨��Ϊnullptr��
	uint64_t byte_offset = 0        // �ֽ�ƫ��
	)
{
	auto dl_tensor = static_cast<DLTensor*>(malloc(sizeof(DLTensor)));
	dl_tensor->data = data;
	dl_tensor->device = device;
	dl_tensor->ndim = ndim;
	dl_tensor->dtype = dtype;
	dl_tensor->byte_offset = byte_offset;

	dl_tensor->shape = static_cast<int64_t*>(malloc(ndim * sizeof(int64_t)));
	if (!dl_tensor->shape)
	{
		free(dl_tensor);
		printf("Failed to allocate memory for shape\n");
		return nullptr;
	}
	memcpy(dl_tensor->shape, shape, ndim * sizeof(int64_t));

	if (strides)
	{
		dl_tensor->strides = static_cast<int64_t*>(malloc(ndim * sizeof(int64_t)));
		if (!dl_tensor->strides)
		{
			free(dl_tensor->shape);
			free(dl_tensor);
			printf("Failed to allocate memory for strides\n");
			return nullptr;
		}
		memcpy(dl_tensor->strides, strides, ndim * sizeof(int64_t));
	}
	else
	{
		dl_tensor->strides = nullptr;
	}

	return dl_tensor;
}

DLPACK_EXTERN_C /*__attribute__ ((visibility("default")))*/ DLManagedTensor* CreateManagedTensor(
    void* data,                     // ����ָ�루���ѷ����ڴ棩
    const DLDevice& device,          // �豸��Ϣ
    const DLDataType& dtype,         // ��������
    int32_t ndim,                    // ά������
    const int64_t* shape,            // ��״����
    const int64_t* strides = nullptr,          // �������飨��Ϊnullptr��
    uint64_t byte_offset = 0,        // �ֽ�ƫ��
    uint64_t flags = 0,              // ��־λ
    bool managedata=true
)
{
	auto dl_managed_tensor = static_cast<DLManagedTensor*>(malloc(sizeof(DLManagedTensor)));
	dl_managed_tensor->dl_tensor.data = data;
	dl_managed_tensor->dl_tensor.device = device;
	dl_managed_tensor->dl_tensor.ndim = ndim;
	dl_managed_tensor->dl_tensor.dtype = dtype;
	dl_managed_tensor->dl_tensor.byte_offset = byte_offset;

	dl_managed_tensor->dl_tensor.shape = static_cast<int64_t*>(malloc(ndim * sizeof(int64_t)));
	if (!dl_managed_tensor->dl_tensor.shape)
	{
		free(dl_managed_tensor);
		printf("Failed to allocate memory for shape\n");
		return nullptr;
	}
	memcpy(dl_managed_tensor->dl_tensor.shape, shape, ndim * sizeof(int64_t));

	if (strides)
	{
		dl_managed_tensor->dl_tensor.strides = static_cast<int64_t*>(malloc(ndim * sizeof(int64_t)));
		if (!dl_managed_tensor->dl_tensor.strides)
		{
			free(dl_managed_tensor->dl_tensor.shape);
			free(dl_managed_tensor);
			printf("Failed to allocate memory for strides\n");
			return nullptr;
		}
		memcpy(dl_managed_tensor->dl_tensor.strides, strides, ndim * sizeof(int64_t));
	}
	else
	{
		dl_managed_tensor->dl_tensor.strides = nullptr;
	}

	dl_managed_tensor->manager_ctx = nullptr;
    if(managedata)
	    dl_managed_tensor->deleter = DLManagedTensorDeleter;
    else
        dl_managed_tensor->deleter = nullptr;
	return dl_managed_tensor;
}

#endif // !DLTENSOR_H
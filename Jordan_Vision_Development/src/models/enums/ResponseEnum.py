from enum import Enum

class ResponseSignal(Enum):
    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    PROCESSING_FILE_SUCCESS = "processing_success"
    PROCESSING_FILE_FAILS = "processing_faild"
    FILE_UPLOAD_SUCCESSFULLY = "file_upload_successfully"
    FILE_UPLOAD_FAILED = "file_upload_failed"

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: image_gen.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'image_gen.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fimage_gen.proto\x12\x08imagegen\"\r\n\x0bPingRequest\"\x1f\n\x0cPingResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\"u\n\x0fGenerateRequest\x12\x0e\n\x06prompt\x18\x01 \x01(\t\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x1b\n\x13num_inference_steps\x18\x04 \x01(\x05\x12\x16\n\x0eguidance_scale\x18\x05 \x01(\x02\"=\n\x10GenerateResponse\x12\x11\n\timage_png\x18\x01 \x01(\x0c\x12\x16\n\x0einference_time\x18\x02 \x01(\x01\x32\x84\x01\n\x08ImageGen\x12\x35\n\x04Ping\x12\x15.imagegen.PingRequest\x1a\x16.imagegen.PingResponse\x12\x41\n\x08Generate\x12\x19.imagegen.GenerateRequest\x1a\x1a.imagegen.GenerateResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'image_gen_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_PINGREQUEST']._serialized_start=29
  _globals['_PINGREQUEST']._serialized_end=42
  _globals['_PINGRESPONSE']._serialized_start=44
  _globals['_PINGRESPONSE']._serialized_end=75
  _globals['_GENERATEREQUEST']._serialized_start=77
  _globals['_GENERATEREQUEST']._serialized_end=194
  _globals['_GENERATERESPONSE']._serialized_start=196
  _globals['_GENERATERESPONSE']._serialized_end=257
  _globals['_IMAGEGEN']._serialized_start=260
  _globals['_IMAGEGEN']._serialized_end=392
# @@protoc_insertion_point(module_scope)

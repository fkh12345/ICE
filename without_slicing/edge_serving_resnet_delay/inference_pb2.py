# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='inference.proto',
  package='base_pakage',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0finference.proto\x12\x0b\x62\x61se_pakage\"9\n\ractionrequest\x12\x0c\n\x04text\x18\x01 \x01(\x0c\x12\r\n\x05start\x18\x02 \x01(\x05\x12\x0b\n\x03\x65nd\x18\x03 \x01(\x05\"=\n\x0e\x61\x63tionresponse\x12\x0c\n\x04text\x18\x01 \x01(\x02\x12\r\n\x05queue\x18\x02 \x01(\x02\x12\x0e\n\x06result\x18\x03 \x01(\x01\x32S\n\nFormatData\x12\x45\n\x08\x44oFormat\x12\x1a.base_pakage.actionrequest\x1a\x1b.base_pakage.actionresponse\"\x00\x62\x06proto3'
)




_ACTIONREQUEST = _descriptor.Descriptor(
  name='actionrequest',
  full_name='base_pakage.actionrequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='base_pakage.actionrequest.text', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='start', full_name='base_pakage.actionrequest.start', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='end', full_name='base_pakage.actionrequest.end', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=32,
  serialized_end=89,
)


_ACTIONRESPONSE = _descriptor.Descriptor(
  name='actionresponse',
  full_name='base_pakage.actionresponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='base_pakage.actionresponse.text', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='queue', full_name='base_pakage.actionresponse.queue', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='result', full_name='base_pakage.actionresponse.result', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=91,
  serialized_end=152,
)

DESCRIPTOR.message_types_by_name['actionrequest'] = _ACTIONREQUEST
DESCRIPTOR.message_types_by_name['actionresponse'] = _ACTIONRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

actionrequest = _reflection.GeneratedProtocolMessageType('actionrequest', (_message.Message,), {
  'DESCRIPTOR' : _ACTIONREQUEST,
  '__module__' : 'inference_pb2'
  # @@protoc_insertion_point(class_scope:base_pakage.actionrequest)
  })
_sym_db.RegisterMessage(actionrequest)

actionresponse = _reflection.GeneratedProtocolMessageType('actionresponse', (_message.Message,), {
  'DESCRIPTOR' : _ACTIONRESPONSE,
  '__module__' : 'inference_pb2'
  # @@protoc_insertion_point(class_scope:base_pakage.actionresponse)
  })
_sym_db.RegisterMessage(actionresponse)



_FORMATDATA = _descriptor.ServiceDescriptor(
  name='FormatData',
  full_name='base_pakage.FormatData',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=154,
  serialized_end=237,
  methods=[
  _descriptor.MethodDescriptor(
    name='DoFormat',
    full_name='base_pakage.FormatData.DoFormat',
    index=0,
    containing_service=None,
    input_type=_ACTIONREQUEST,
    output_type=_ACTIONRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_FORMATDATA)

DESCRIPTOR.services_by_name['FormatData'] = _FORMATDATA

# @@protoc_insertion_point(module_scope)

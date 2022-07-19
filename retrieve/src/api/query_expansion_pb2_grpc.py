# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from api import query_expansion_pb2 as api_dot_query__expansion__pb2


class ExpandStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Expand = channel.unary_unary(
                '/api.Expand/Expand',
                request_serializer=api_dot_query__expansion__pb2.ExpandRequest.SerializeToString,
                response_deserializer=api_dot_query__expansion__pb2.ExpandResponse.FromString,
                )


class ExpandServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Expand(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ExpandServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Expand': grpc.unary_unary_rpc_method_handler(
                    servicer.Expand,
                    request_deserializer=api_dot_query__expansion__pb2.ExpandRequest.FromString,
                    response_serializer=api_dot_query__expansion__pb2.ExpandResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'api.Expand', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Expand(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Expand(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/api.Expand/Expand',
            api_dot_query__expansion__pb2.ExpandRequest.SerializeToString,
            api_dot_query__expansion__pb2.ExpandResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
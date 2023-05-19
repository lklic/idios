import json
import pika
import uuid
import os
from common import JOB_QUEUE_NAME


class RpcClient(object):
    def __init__(self, queue_name=JOB_QUEUE_NAME):
        self.queue_name = queue_name
        self.connection = pika.BlockingConnection(
            pika.URLParameters(
                os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672")
            )
        )

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue="", exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True,
        )

        self.response_data = None
        self.corr_id = None
        self.valid = True

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response_data = body

    def __call__(self, command, args):
        self.response_data = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps([command, args]),
        )
        self.connection.process_data_events(time_limit=10)  # seconds
        if self.response_data is None:
            raise RuntimeError("No response (timeout?)")
        response = json.loads(self.response_data)
        if (
            hasattr(response, "__contains__")
            and "exception_type" in response
            and "exception_args" in response
        ):
            exception_type = response["exception_type"]
            exception_args = response["exception_args"]
            exec(f"raise {exception_type}(*{exception_args})")
        return response

from aws_cdk import (
    Stack,
    aws_lambda as _lambda,
    aws_apigateway as apigateway,
)
from constructs import Construct
from aws_cdk import Duration

class CdkDockerLambdaStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        self.prediction_lambda = _lambda.DockerImageFunction(
            self, 
            "FingerprintProcessingService",
            function_name="FingerprintProcessingService",
            code=_lambda.DockerImageCode.from_image_asset(
                directory="cdk_docker_lambda/ExampleDockerLambda"
            ),
            timeout=Duration.seconds(100),  
            memory_size=1024,  
        )

        api = apigateway.LambdaRestApi(
            self,
            "FingerprintProcessingApiGateway",
            handler=self.prediction_lambda,
            proxy=False,
            default_cors_preflight_options={
                "allow_origins": apigateway.Cors.ALL_ORIGINS,
                "allow_methods": apigateway.Cors.ALL_METHODS,
                "allow_headers": ["Content-Type"],
            },
        )

        process = api.root.add_resource("process")
        extract = process.add_resource("extract")
        extract.add_method("POST")
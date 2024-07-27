from aws_cdk import App
from cdk_docker_lambda.cdk_docker_lambda_stack import CdkDockerLambdaStack

app = App()
CdkDockerLambdaStack(app, "CdkDockerLambdaStack")
app.synth()
import os

import s3fs


class S3FileSystem(s3fs.S3FileSystem):
    def __init__(
            self,
            endpoint_env_var: str = "AWS_S3_ENDPOINT",
            key_env_var: str = "AWS_ACCESS_KEY_ID",
            secret_env_var: str = "AWS_SECRET_ACCESS_KEY",
            token_env_var: str = "AWS_SESSION_TOKEN",
    ):
        super().__init__(
            client_kwargs={"endpoint_url": "https://" + os.environ[endpoint_env_var]},
            key=os.environ[key_env_var],
            secret=os.environ[secret_env_var],
            token=os.environ[token_env_var]
        )

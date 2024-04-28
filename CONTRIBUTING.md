# Code and Contribution guidelines

## Code formatting

Formatting is enforced using [yapf](https://github.com/google/yapf) and automatically applied using pre-commit hooks. To manually format the code, run the following command:

```shell
yapf -i -r .
```
but we highly recommend using the pre-commit hooks to ensure consistent formatting across the codebase, see below.

### Pre-commit

We use pre-commit to enforce code formatting and quality standards. Follow these steps to install and use pre-commit:

1. Make sure you have Python installed on your system.

2. Install pre-commit by running the following command in your terminal:

    ```shell
    pip install pre-commit
    ```

3. Navigate to the root directory of your project.

4. Run the following command to initialize pre-commit:

    ```shell
    pre-commit install
    ```

5. Now, whenever you make a commit, pre-commit will automatically run the configured hooks on the files you modified. If any issues are found, pre-commit will prevent the commit from being made.

    You can also manually run pre-commit on all files by running the following command:

    ```shell
    pre-commit run --all-files
    ```

    This is useful if you want to check all files before making a commit.

6. Customize the pre-commit configuration by creating a `.pre-commit-config.yaml` file in the root directory of your project. This file allows you to specify which hooks should be run and how they should be configured.

    For more information on configuring pre-commit, refer to the [pre-commit documentation](https://pre-commit.com/#configuration).

That's it! You now have pre-commit set up to automatically enforce code formatting and quality standards in your project.

from triton.tools.env import getTritonBasePath
from .sim_arguments import FFMConfig
import os
import subprocess


def create_config_id(config):
    config_key_value_pairs = [f"{key}_{value}" for key, value in config.items()]
    return "_".join(config_key_value_pairs)


def create_output_dir(config, prefix):
    return os.path.join(getTritonBasePath(), prefix, create_config_id(config))


def generateFFMConfigs(ffmConfigs: list[FFMConfig]):
    config_str = ""
    for config in ffmConfigs:
        prefix = config.name
        test_name = f"{prefix}_test"
        if config.id:
            test_name = f"{prefix}_test_{create_config_id(config.id)}"
        config_str += f"""
        {{
                "{test_name}",
                "{config.sp3}",
                "{config.regIni}",
                "{config.surfaceIni}",
                {config.variance}f
        }},
        """
    return config_str


def generateAMConfigs(ffmConfigs: list[FFMConfig]):
    config_str = ""
    for config in ffmConfigs:
        prefix = config.name
        test_name = f"CS_PM4({prefix}_test"
        if config.id:
            test_name = f"{prefix}_test_{create_config_id(config.id)}"
        config_str += f"""
        CS_PM4({test_name},
            test.ini.test_args.wave64_mode=0;
            test.ini.test_args.pm4lib_CS_W32_EN=1;
            test.ini.test_args.LS={config.sp3};
            test.ini.test_args.ini={config.surfaceIni};
            test.ini.test_args.ini_1={config.regIni};
            test.ini.test_args.tc_FBLocation=0x00007F8550000000;
            test.ini.test_args.Varience={config.variance};
        )"""


#CS_PM4(flash_attention_giuseppe,
#    test.ini.test_args.gen=cs_pm4;
#    test.ini.test_args.wave64_mode=0;
#    test.ini.test_args.pm4lib_CS_W32_EN=1;
#    test.ini.test_args.LS=/proj/hpc_solutions/Negar/triton_flash_attention/run/flash_attention/flash_attention_mi400.sp3
#    test.ini.test_args.ini=/proj/hpc_solutions/Negar/triton_flash_attention/run/flash_attention/flash_attention_mi400_me
#    test.ini.test_args.ini_1=/proj/hpc_solutions/Negar/triton_flash_attention/run/flash_attention/flash_attention_mi400_
#    test.ini.test_args.tc_FBLocation=0x00007F8550000000;
#    test.ini.test_args.Varience=0.01;
#)
    return config_str


def execute_command_in_folder(command, folder_path):
    try:
        process = subprocess.Popen(command, cwd=folder_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"Command executed successfully in {folder_path}")
        else:
            print(f"Command failed in {folder_path} with error code {process.returncode}")

        return (stdout.decode(), stderr.decode())
    except FileNotFoundError:
        print(f"Error: Folder not found: {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def getPtype(dtype):
    ptype = "fp16"
    if dtype == "bfloat16":
        ptype = "bf16"
    elif dtype == "float8_e4m3fn":
        ptype = "fp8e4nv"
    elif dtype == "float8_e5m2":
        ptype = "fp8e5"
    return ptype


def generateNiceTable(headers, data, title):
    """
    Creates an HTML table from a list of lists.

    Args:
        data: A list of lists representing the table data.
        headers: An optional list of strings representing the table headers.

    Returns:
        A string containing the HTML table.
    """

    html_table = "<section>\n"
    html_table += f"<h2> {title} </h2>\n"
    html_table += "<table>\n"

    if headers:
        html_table += "  <tr>\n"
        for header in headers:
            html_table += f"    <th>{header}</th>\n"
        html_table += "  </tr>\n"

    for row in data:
        html_table += "  <tr>\n"
        for cell in row:
            html_table += f"    <td>{cell}</td>\n"
        html_table += "  </tr>\n"

    html_table += "</table>\n"
    html_table += "</section>\n"
    return html_table


def createHtmlPage(title="My Web Page", content="<h1>Hello, World!</h1>"):
    """Creates a basic HTML file.

    Args:
        filename: The name of the HTML file to create (default: index.html).
        title: The title of the HTML page (default: My Web Page).
        content: The HTML content of the page (default: <h1>Hello, World!</h1>).
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #121212;
                color: #f0f0f0;
            }}
            section {{
                background-color: #333;
                padding: 20px;
                border: 1px solid #444;
                border-radius: 8px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                background-color: #222;
            }}
            th, td {{
                border: 1px solid #555;
                padding: 10px;
                text-align: left;
                color: #ddd;
            }}
            th {{
                background-color: #444;
                color: #fff;
            }}
            tr:nth-child(even) {{
                background-color: #2a2a2a;
            }}
            tr:hover {{
                background-color: #444;
            }}
        </style>
    </head>
    <body>
        {content}
    </body>
    </html>
    """
    return html_content

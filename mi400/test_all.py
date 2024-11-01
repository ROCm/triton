from sim.aot import aot_compile
from sim.mi400Simulator import MI400Simulator
from sim.sim_arguments import Arguments, FFMConfig
from sim.utils import *
import test_fa
import test_gemm
import test_softmax
import os

testers = [test_gemm, test_fa, test_softmax]

if __name__ == "__main__":
    # Generate all the configurations we care about
    cfgstr = ""
    for t in testers:
        ffmConfigs = t.testAllConfigs()
        cfgstr += (generateFFMConfigs(ffmConfigs=ffmConfigs))
    ffm = os.getenv("FFM_PATH")
    ffmBin = os.getenv("FFM_BIN_PATH")

    # Dump the configuration into the header test file
    f = open(os.path.join(ffm, "test/shader/cs_pm4/cs_pm4_tests.h"), "r")
    lines = []
    for line in f:
        line = line.strip()
        if "cs_pm4_tests[] = {" in line:
            lines.append(line)
            for c in cfgstr.split("\n"):
                lines.append(c.strip())
            lines.append("};")
            break
        else:
            lines.append(line)
    f.close()

    f = open(os.path.join(ffm, "test/shader/cs_pm4/cs_pm4_tests.h"), "w")
    for l in lines:
        print(l, file=f)
    f.close()

    # Build
    results, _ = execute_command_in_folder("conan build ..", os.path.join(ffmBin, ".."))
    print(results)

    # Run and print if the test failed or worked
    results, errors = execute_command_in_folder("cs_pm4_tests", ffmBin)
    verifyRun = False
    ffmResults = {}
    configStr = ""
    for r in results.split("\n"):
        if "CsPm4Test.Run" in r:
            if "RUN" in r:
                verifyRun = True
                configStr = r.split("/")[-1]
                continue
            if verifyRun:
                if "OK" in r:
                    ffmResults[configStr] = "Passed"
                else:
                    ffmResults[configStr] = "Failed"
                verifyRun = False
        elif verifyRun:
            ffmResults[configStr] = "Failed"

    html_table = ""
    for t in testers:
        headers, data, title = t.getHeadersAndData(ffmResults)
        if len(data):
            html_table += generateNiceTable(headers, data, title)

    summary = createHtmlPage("Test results", html_table)
    # To save to an HTML file:
    with open("MI400Status.html", "w") as f:
        f.write(summary)

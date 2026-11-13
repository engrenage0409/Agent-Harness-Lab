import argparse,json
from coding.test_runner import run_tests

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workspace",default="workspace")
    p.add_argument("--approve",action="store_true",help="I trust the workspace code and consent to execute tests")
    p.add_argument("--timeout",type=int,default=30)
    a=p.parse_args()
    print(json.dumps(run_tests(a.workspace,approved=a.approve,timeout=a.timeout),indent=2))

if __name__=="__main__":main()

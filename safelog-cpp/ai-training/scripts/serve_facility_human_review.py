"""Loopback-only, auto-saving human review; serves selected assets only."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys
import threading
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.review_feedback import validate_package,validate_feedback,read_json,TASKS,canonical_sha256
from safelog_ai.human_training_feedback import SCHEMA,validate_submission
from scripts.train_facility_target import sha

MAX_BYTES=2_000_000


def require(condition,message):
    if not condition:raise ValueError(message)


def strict_json(raw):
    def pairs(values):
        result={}
        for k,v in values:
            if k in result:raise ValueError("Duplicate JSON key")
            result[k]=v
        return result
    def bad(value):raise ValueError("Nonfinite JSON value")
    return json.loads(raw.decode("utf-8"),object_pairs_hook=pairs,parse_constant=bad)


def blank_draft(package):
    return {"source_package_sha256":package["package_content_sha256"],"weights_sha256":package["weights_sha256"],
        "reviewer":{"reviewer_id":"local-reviewer-1","name":"검수자1"},
        "cases":[{"case_id":c["case_id"],"judgements":["unreviewed","unreviewed"],"note":"","reviewed_at":[None,None]}for c in package["cases"]]}


def to_feedback(draft,package):
    require(isinstance(draft,dict)and set(draft)=={"source_package_sha256","weights_sha256","reviewer","cases"},"입력 형식이 잘못되었습니다.")
    require(isinstance(draft["reviewer"],dict)and set(draft["reviewer"])=={"reviewer_id","name"},"검수자 입력을 확인하세요.")
    require(isinstance(draft["cases"],list)and len(draft["cases"])==len(package["cases"]),"사진 수가 맞지 않습니다.")
    require([c.get("case_id")for c in draft["cases"]if isinstance(c,dict)]==[c["case_id"]for c in package["cases"]],"사진의 순서나 번호가 맞지 않습니다.")
    rows=[];labels={"concrete_crack":"균열","concrete_spalling":"박락"};words={"present":"있음","absent":"없음","uncertain":"판단 불가"}
    for case in draft["cases"]:
        require(isinstance(case,dict)and set(case)=={"case_id","judgements","note","reviewed_at"},"사진 입력 형식이 잘못되었습니다.")
        require(isinstance(case["note"],str)and len(case["note"])<=3000,"메모는3000자 이하로 입력하세요.")
        require(isinstance(case["judgements"],list)and len(case["judgements"])==2 and isinstance(case["reviewed_at"],list)and len(case["reviewed_at"])==2,"균열·박락 판단을 확인하세요.")
        tasks=[]
        for k,task in enumerate(TASKS):
            judgement=case["judgements"][k]
            require(judgement in("unreviewed","present","absent","uncertain"),"손상 판단을 확인하세요.")
            if judgement=="unreviewed":
                require(case["reviewed_at"][k]is None,"검수 전 항목의 날짜를 확인하세요.")
                tasks.append({"task":task,"judgement":judgement,"reason":"unreviewed","note":"","evidence":"","reviewed_at":None})
            else:
                note=f"직접 선택한 관찰: {labels[task]} {words[judgement]}"
                if case["note"].strip():note+=". "+case["note"].strip()
                tasks.append({"task":task,"judgement":judgement,"reason":"annotation_uncertain"if judgement=="uncertain"else"other",
                              "note":note,"evidence":"","reviewed_at":case["reviewed_at"][k]})
        rows.append({"case_id":case["case_id"],"tasks":tasks})
    doc={"schema":"facility_train_review_feedback_v2","source_package_sha256":draft["source_package_sha256"],
         "weights_sha256":draft["weights_sha256"],"exported_utc":datetime.now(timezone.utc).isoformat(),
         "reviewer":{**draft["reviewer"],"role":"team_observer","expertise":""},"cases":rows}
    return validate_feedback(doc,package)


def counts(draft):
    return {"photos":len(draft["cases"]),"complete_photos":sum(all(v!="unreviewed"for v in c["judgements"])for c in draft["cases"]),
            "reviewed_tasks":sum(v!="unreviewed"for c in draft["cases"]for v in c["judgements"]),
            "definite_tasks":sum(v in("present","absent")for c in draft["cases"]for v in c["judgements"]),
            "uncertain_tasks":sum(v=="uncertain"for c in draft["cases"]for v in c["judgements"])}


class ReviewStore:
    def __init__(self,package_path,*,test_fixture=False):
        self.package_path=Path(package_path).resolve();self.directory=self.package_path.parent
        self.package=validate_package(read_json(self.package_path));self.token=secrets.token_urlsafe(24)
        self.test_fixture=test_fixture;self.lock=threading.Lock();self.assets={}
        for case in self.package["cases"]:
            image=(ROOT/case["image"]).resolve()
            require(image.is_relative_to(ROOT/"data")and image.is_file()and sha(image)==case["image_sha256"],"검수 사진 파일을 확인하세요.")
            self.assets["/photo/"+case["case_id"]]=image
        self.draft_path=self.directory/"human-draft.json"
        self.record=read_json(self.draft_path)if self.draft_path.exists()else{"revision":0,"draft":blank_draft(self.package)}
        to_feedback(self.record["draft"],self.package)
    def save(self,revision,draft):
        to_feedback(draft,self.package)
        with self.lock:
            require(type(revision)is int and revision==self.record["revision"],"다른 창에서 입력이 바뀌었습니다. 현재 입력을 파일로 보관하고 새로고침하세요.")
            record={"revision":revision+1,"draft":draft}
            temp=self.draft_path.with_suffix(".pending.json")
            with temp.open("w",encoding="utf-8",newline="\n")as stream:stream.write(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
            temp.replace(self.draft_path);self.record=record
            return {"revision":record["revision"],"counts":counts(draft)}
    def submit(self,revision):
        with self.lock:
            require(type(revision)is int and revision==self.record["revision"],"저장 후 다시 제출하세요.")
            feedback=to_feedback(self.record["draft"],self.package)
            submission={"schema":SCHEMA,"purpose":"human_reviewed_train_photo_labels",
                "source_package_sha256":self.package["package_content_sha256"],"feedback":feedback,
                "submitted_for_training":True,"test_fixture":self.test_fixture}
            if not self.test_fixture:validate_submission(submission,self.package)
            folder=self.directory/"submissions";folder.mkdir(exist_ok=True)
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target=folder/("HUMAN-TRAINING-"+stamp+".json")
            with target.open("x",encoding="utf-8",newline="\n")as stream:stream.write(json.dumps(submission,ensure_ascii=False,indent=2)+"\n")
            return {"status":"submitted","filename":target.name,"revision":revision,"counts":counts(self.record["draft"]),"training_started":False,"test_fixture":self.test_fixture}


class Handler(BaseHTTPRequestHandler):
    server_version="SafeLogHumanReview/1"
    def log_message(self,format,*args):pass
    def send(self,code,body,mime="application/json; charset=utf-8"):
        if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode("utf-8")
        self.send_response(code);self.send_header("Content-Type",mime);self.send_header("Content-Length",str(len(body)))
        self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff")
        self.send_header("Content-Security-Policy","default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(body)
    def local_host(self):
        return self.headers.get("Host")=="127.0.0.1:"+str(self.server.server_port)
    def do_GET(self):
        if not self.local_host():return self.send(403,{"error":"Localhost access only"})
        path=urlsplit(self.path).path;store=self.server.store
        if path=="/api/package":
            return self.send(200,{"token":store.token,"test_fixture":store.test_fixture,
                "package_sha256":store.package["package_content_sha256"],"cases":[{"case_id":c["case_id"],"domain":c["domain"],
                    "image_url":"/photo/"+c["case_id"],"original_targets":c["original_targets"][:2],"probabilities":c["probabilities"][:2],
                    "thresholds":[store.package["thresholds"][t]for t in TASKS]}for c in store.package["cases"]]})
        if path=="/api/draft":
            with store.lock:return self.send(200,{**store.record,"counts":counts(store.record["draft"])})
        if path=="/api/export":
            with store.lock:return self.send(200,{"schema":SCHEMA,"purpose":"human_reviewed_train_photo_labels","source_package_sha256":store.package["package_content_sha256"],
                "feedback":to_feedback(store.record["draft"],store.package),"submitted_for_training":False,"test_fixture":store.test_fixture})
        if path in store.assets:return self.send(200,store.assets[path].read_bytes(),"image/jpeg")
        static={"/":"human-review.html","/human-review.js":"human-review.js","/human-review.css":"human-review.css"}
        if path in static:
            mime={"html":"text/html; charset=utf-8","js":"text/javascript; charset=utf-8","css":"text/css; charset=utf-8"}[static[path].rsplit(".",1)[1]]
            return self.send(200,(ROOT/"review-ui"/static[path]).read_bytes(),mime)
        return self.send(404,{"error":"Not found"})
    def do_POST(self):
        store=self.server.store;origin="http://127.0.0.1:"+str(self.server.server_port)
        if not self.local_host()or self.headers.get("Origin")not in(None,origin)or self.headers.get("X-Review-Token")!=store.token:
            return self.send(403,{"error":"Local review session required"})
        try:
            length=int(self.headers.get("Content-Length","0"));require(0<length<=MAX_BYTES,"Input is too large")
            doc=strict_json(self.rfile.read(length));path=urlsplit(self.path).path
            if path=="/api/draft":
                require(isinstance(doc,dict)and set(doc)=={"revision","draft"},"Invalid draft request")
                return self.send(200,store.save(doc["revision"],doc["draft"]))
            if path=="/api/submit":
                require(isinstance(doc,dict)and set(doc)=={"revision"},"Invalid submission request")
                return self.send(200,store.submit(doc["revision"]))
            return self.send(404,{"error":"Not found"})
        except(ValueError,KeyError,TypeError,UnicodeError)as error:return self.send(400,{"error":str(error)})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--package",type=Path,required=True)
    parser.add_argument("--port",type=int,default=8770);parser.add_argument("--test-fixture",action="store_true");args=parser.parse_args()
    path=args.package.resolve();require(path.is_relative_to((ROOT/"runs").resolve()),"Review data must be in local ignored runs")
    store=ReviewStore(path,test_fixture=args.test_fixture)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler);server.store=store
    print(json.dumps({"url":f"http://127.0.0.1:{server.server_port}/","photos":len(store.package["cases"]),"test_fixture":args.test_fixture}),flush=True)
    server.serve_forever()


if __name__=="__main__":main()

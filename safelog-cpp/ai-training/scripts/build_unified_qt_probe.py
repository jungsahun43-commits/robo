"""Build the actual modified Qt adapter tests and HTTP probe in an isolated directory."""
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
BASE=ROOT/'runs/unified-main-qt-probe'
if __name__=='__main__':
    assert not BASE.exists();BASE.mkdir()
    app=REPO/'safelog-cpp/apps/mobile-qt';include=REPO/'safelog-cpp/src/ai/include'
    text=f'''cmake_minimum_required(VERSION 3.24)
project(UnifiedMainProbe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_AUTOMOC ON)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Network Test)
add_executable(adapter_tests "{(app/'tests/http_ai_safety_analyzer_tests.cpp').as_posix()}" "{(app/'adapters/ai/http_ai_safety_analyzer.cpp').as_posix()}")
target_include_directories(adapter_tests PRIVATE "{app.as_posix()}" "{include.as_posix()}")
target_link_libraries(adapter_tests PRIVATE Qt6::Core Qt6::Gui Qt6::Network Qt6::Test)
add_executable(unified_probe "{(app/'tools/unified_main_probe.cpp').as_posix()}" "{(app/'adapters/ai/http_ai_safety_analyzer.cpp').as_posix()}")
target_include_directories(unified_probe PRIVATE "{app.as_posix()}" "{include.as_posix()}")
target_link_libraries(unified_probe PRIVATE Qt6::Core Qt6::Network)
'''
    (BASE/'CMakeLists.txt').write_text(text,encoding='utf-8')
    env=dict(os.environ);env['PATH']='C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH','');env['QT_QPA_PLATFORM']='offscreen'
    cmake='C:/Qt/Tools/CMake_64/bin/cmake.exe'
    subprocess.run([cmake,'-S',str(BASE),'-B',str(BASE/'build'),'-G','Ninja',
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH=C:/Qt/6.11.2/mingw_64',
        '-DCMAKE_MAKE_PROGRAM=C:/Qt/Tools/Ninja/ninja.exe','-DCMAKE_CXX_COMPILER=C:/Qt/Tools/mingw1310_64/bin/g++.exe'],env=env,check=True)
    subprocess.run([cmake,'--build',str(BASE/'build'),'-j','4'],env=env,check=True)
    result=subprocess.run([str(BASE/'build/adapter_tests.exe')],env=env,check=True,capture_output=True,text=True)
    (BASE/'adapter-test-output.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    report={'status':'passed','actual_modified_Qt_CPP_adapter_compiled':True,'adapter_test_output':result.stdout.strip(),
        'probe_executable':str(BASE/'build/unified_probe.exe'),'compiler':'Qt6.11.2 MinGW13.1 C++20',
        'full_Android_APK_built':False}
    with (ROOT/'reports/unified-main-qt-build.json').open('x',encoding='utf-8') as out:json.dump(report,out,ensure_ascii=False,indent=2)
    print(result.stdout,flush=True)

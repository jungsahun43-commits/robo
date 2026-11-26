import { Fragment, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { router } from 'expo-router';
import { Button, Card, Copy, Field, message, Page } from '@/components/safelog-ui';
import { isPcbTask } from '@/contracts/inspection';
import type { AiMode, InspectionTask, Photo, Suggestion } from '@/contracts/types';
import { aiConnection } from '@/features/ai/connection-store';
import { createFixedCpuAnalyzer } from '@/features/ai/fixed-cpu';
import { readPhotoDataUrl } from '@/features/ai/photo-data';
import { routePhoto } from '@/features/ai/photo-routing';
import { useSession, workflow } from '@/features/auth/session';
import { site } from '@/features/auth/demo';
import { PhotoField } from '@/features/capture/photo-field';
import { demoSelectionNote, demoTasks, inspectionTasks, photoRatesLabel, taskGroups, taskOption } from '@/features/ai/tasks';

type InputMode = 'auto' | 'choose' | 'manual' | 'mock';
type AutoResult = { photo: Photo; revision: number; attempt: number } & (
  { suggestion: Suggestion; error?: never } | { suggestion?: never; error: string }
);

export default function FindingCapturePage() {
  const { user } = useSession();
  const connection = useSyncExternalStore(aiConnection.subscribe, aiConnection.getSnapshot, aiConnection.getSnapshot);
  const submitted = useRef(false);
  const request = useRef<AbortController | null>(null);
  const [photo, setPhoto] = useState<Photo>();
  const [mode, setMode] = useState<InputMode>('auto');
  const [manualTask, setManualTask] = useState<InspectionTask>();
  const [autoResult, setAutoResult] = useState<AutoResult>();
  const [attempt, setAttempt] = useState(0);
  const [onlyDemo, setOnlyDemo] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [location, setLocation] = useState('');
  const [description, setDescription] = useState('');
  const [action, setAction] = useState('');
  const [error, setError] = useState('');

  const result = mode === 'auto' && autoResult && autoResult.photo === photo
    && autoResult.revision === connection.revision && autoResult.attempt === attempt ? autoResult : undefined;
  const inspectionTask = mode === 'auto' ? result?.suggestion?.inspectionTask
    : mode === 'choose' ? manualTask : mode === 'manual' ? 'manual_other' : undefined;
  const aiMode: AiMode = mode === 'auto' || mode === 'choose' ? 'fixed-cpu' : mode;

  useEffect(() => {
    if (mode !== 'auto' || !photo) return;
    const controller = new AbortController();
    request.current = controller;
    const { signal } = controller;
    const revision = connection.revision;
    const isCurrent = () => !signal.aborted && request.current === controller
      && aiConnection.getSnapshot().revision === revision;

    async function analyze(input: Photo) {
      try {
        const task = await routePhoto(input, { connection, readPhoto: readPhotoDataUrl, signal });
        if (!isCurrent()) return;
        if (!task) throw new Error('사진에서 점검 대상을 확실하게 찾지 못했습니다.');
        const analyzer = createFixedCpuAnalyzer({ ...connection, task,
          readPhoto: (selected, active) => readPhotoDataUrl(selected, active, isPcbTask(task) ? 40_000_000 : 12_000_000),
        });
        const suggestion = await analyzer.hazard(input, signal);
        if (isCurrent()) setAutoResult({ photo: input, revision, attempt, suggestion });
      } catch (error) {
        if (isCurrent()) setAutoResult({ photo: input, revision, attempt, error: message(error) });
      } finally {
        if (request.current === controller) request.current = null;
      }
    }

    void analyze(photo);
    return () => controller.abort();
  }, [attempt, connection, mode, photo]);

  function changePhoto(next: Photo) {
    request.current?.abort();
    setAutoResult(undefined);
    setManualTask(undefined);
    setError('');
    setPhoto(next);
  }

  function changeMode(next: InputMode) {
    if (next === mode) return;
    request.current?.abort();
    setAutoResult(undefined);
    setError('');
    setMode(next);
  }

  function register() {
    if (submitted.current) return;
    submitted.current = true;
    try {
      if (!user || !photo) throw new Error('점검 사진을 먼저 선택하세요.');
      if (mode === 'auto' && (!result?.suggestion || result.revision !== aiConnection.getSnapshot().revision)) {
        throw new Error('현재 사진의 자동 검사 결과를 확인하세요.');
      }
      if (mode === 'choose' && !inspectionTask) throw new Error('점검 분야를 선택하세요.');
      const finding = workflow.create(user, { photo, location, description, action, aiMode,
        inspectionTask }, result?.suggestion);
      router.replace({ pathname: '/hazard', params: mode === 'auto' ? { id: finding.id } : { id: finding.id, fresh: '1' } });
    } catch (error) {
      submitted.current = false;
      setError(message(error));
    }
  }

  return <Page title="안전점검 등록">
    <Card>
      <Copy>점검 대상 · {site.name}</Copy>
      <Button title="현장 정보 보기" secondary onPress={() => router.push("/facilities")} />
    </Card>
    <Card><PhotoField label="조치 전 사진" photo={photo} onChange={changePhoto} /></Card>
    <Card>
      {mode === 'auto' && <>
        {!photo && <Copy>사진을 선택하면 연결된 AI 서버에서 점검 대상과 손상 후보를 확인합니다.</Copy>}
        {photo && !result && <Copy tone="ai">점검 대상과 손상 후보를 확인하는 중입니다.</Copy>}
        {result?.suggestion && <>
          <Copy tone="ai">자동 판별 · {taskOption(inspectionTask)?.label}</Copy>
          <Copy>{result.suggestion.candidateDetected ? '손상 후보 발견' : '손상 후보 미검출'}</Copy>
          <Copy>{result.suggestion.assessment}</Copy>
          <Copy>{result.suggestion.action}</Copy>
        </>}
        {result?.error && <>
          <Copy tone="error">{result.error}</Copy>
          <Button title="자동 검사 다시 시도" onPress={() => setAttempt(value => value + 1)} />
        </>}
      </>}
      {mode === 'auto' && !result?.error && <Button title={showOptions ? '다른 검사 방법 접기' : '다른 검사 방법'} secondary
        onPress={() => setShowOptions(value => !value)} />}
      {(showOptions || mode !== 'auto' || result?.error) && <>
        {mode !== 'auto' && <Button title="자동 검사 사용" secondary onPress={() => {
          setShowOptions(false);
          changeMode('auto');
        }} />}
        {mode !== 'choose' && <Button title="점검 분야 직접 선택" secondary onPress={() => changeMode('choose')} />}
        {mode !== 'manual' && <Button title="AI 없이 직접 검토" secondary onPress={() => changeMode('manual')} />}
        {mode !== 'mock' && <Button title="모의 시연(Mock)" secondary onPress={() => changeMode('mock')} />}
      </>}
      {mode === 'choose' && <>
        <Copy>사진에 맞는 점검 분야를 선택하세요.</Copy>
        <Button title={onlyDemo ? '전체 분야 보기' : '전체 분야 · 선택됨'} secondary={onlyDemo} onPress={() => setOnlyDemo(false)} />
        <Button title={onlyDemo ? '발표 선정 5개 · 선택됨' : '발표 선정 5개만 보기'} secondary={!onlyDemo} onPress={() => setOnlyDemo(true)} />
        {onlyDemo && <Copy>{demoSelectionNote}</Copy>}
        {(onlyDemo ? ['발표 선정'] : taskGroups).map(group => <Fragment key={group}>
          <Copy>{group}</Copy>
          {(onlyDemo ? demoTasks : inspectionTasks.filter(task => task.group === group)).map(task => <Button key={task.id}
            title={(task.id === inspectionTask ? '선택됨 · ' : '') + task.label}
            secondary={task.id !== inspectionTask} onPress={() => {
              if (task.id === 'manual_other') changeMode('manual');
              else setManualTask(task.id);
            }} />)}
        </Fragment>)}
        {inspectionTask && <>
          <Copy>{taskOption(inspectionTask)?.scope}</Copy>
          <Copy>{taskOption(inspectionTask)?.evaluation} · {photoRatesLabel(inspectionTask)}</Copy>
        </>}
      </>}
      {mode === 'manual' && <Copy>이 사진을 직접 검토 모드로 등록합니다.</Copy>}
      {mode === 'mock' && <Copy>모의 시연 모드입니다.</Copy>}
    </Card>
    <Card>
      <Field label="점검 위치" value={location} onChangeText={setLocation} />
      <Field label="발견 메모" value={description} onChangeText={setDescription} multiline />
      <Field label="수동 권장 조치" value={action} onChangeText={setAction} multiline />
    </Card>
    {!!error && <Copy tone="error">{error}</Copy>}
    <Copy>등록하면 미조치 상태로 저장되고 {aiMode === 'manual' ? '직접 검토' : 'AI 검토'}가 열립니다.</Copy>
    <Button title={aiMode === 'manual' ? '등록 후 직접 검토' : '등록 후 AI 검토'} onPress={register}
      disabled={!photo || (aiMode === 'fixed-cpu' && !inspectionTask)} />
  </Page>;
}

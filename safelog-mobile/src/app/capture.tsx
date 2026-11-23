import { Fragment, useRef, useState } from 'react';
import { router } from 'expo-router';
import { Button, Card, Copy, Field, message, Page } from '@/components/safelog-ui';
import type { AiMode, InspectionTask, Photo } from '@/contracts/types';
import { useSession, workflow } from '@/features/auth/session';
import { site } from '@/features/auth/demo';
import { PhotoField } from '@/features/capture/photo-field';
import { demoSelectionNote, demoTasks, inspectionTasks, isDemoTask, photoRatesLabel, taskGroups, taskOption } from '@/features/ai/tasks';

export default function FindingCapturePage() {
  const { user } = useSession();
  const submitted = useRef(false);
  const [photo, setPhoto] = useState<Photo>();
  const [aiMode, setAiMode] = useState<AiMode>('fixed-cpu');
  const [inspectionTask, setInspectionTask] = useState<InspectionTask>('facility_surface');
  const [onlyDemo, setOnlyDemo] = useState(false);
  const [location, setLocation] = useState('');
  const [description, setDescription] = useState('');
  const [action, setAction] = useState('');
  const [error, setError] = useState('');

  function register() {
    if (submitted.current) return;
    submitted.current = true;
    try {
      if (!user || !photo) throw new Error('점검 사진을 먼저 선택하세요.');
      const finding = workflow.create(user, { photo, location, description, action, aiMode,
        inspectionTask: aiMode === 'mock' ? undefined : inspectionTask });
      router.replace({ pathname: '/hazard', params: { id: finding.id, fresh: '1' } });
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
    <Card>
      <Copy>AI 분석 방식</Copy>
      <Button title={(aiMode === 'fixed-cpu' ? '선택됨 · ' : '') + '기존 고정 AI 모델'} secondary={aiMode !== 'fixed-cpu'}
        onPress={() => { setAiMode('fixed-cpu'); if (inspectionTask === 'manual_other') setInspectionTask('facility_surface'); }} />
      <Button title={(aiMode === 'mock' ? '선택됨 · ' : '') + '모의 시연(Mock)'} secondary={aiMode !== 'mock'}
        onPress={() => setAiMode('mock')} />
      <Button title={(aiMode === 'manual' ? '선택됨 · ' : '') + '기타 분야 직접 검토'} secondary={aiMode !== 'manual'}
        onPress={() => { setAiMode('manual'); setInspectionTask('manual_other'); setOnlyDemo(false); }} />
      {aiMode !== 'mock' ? <>
        <Copy>사진에 맞는 점검 분야를 선택하세요.</Copy>
        <Button title={onlyDemo ? '전체 분야 보기' : '전체 분야 · 선택됨'} secondary={onlyDemo} onPress={() => setOnlyDemo(false)} />
        <Button title={onlyDemo ? '발표 선정 5개 · 선택됨' : '발표 선정 5개만 보기'} secondary={!onlyDemo} onPress={() => {
          setOnlyDemo(true);
          if (!isDemoTask(inspectionTask)) { setInspectionTask(demoTasks[0].id); setAiMode('fixed-cpu'); }
        }} />
        {onlyDemo && <Copy>{demoSelectionNote}</Copy>}
        {(onlyDemo ? ['발표 선정'] : taskGroups).map(group => <Fragment key={group}>
          <Copy>{group}</Copy>
          {(onlyDemo ? demoTasks : inspectionTasks.filter(task => task.group === group)).map(task => <Button key={task.id}
            title={(task.id === inspectionTask ? '선택됨 · ' : '') + task.label}
            secondary={task.id !== inspectionTask} onPress={() => {
              setInspectionTask(task.id); setAiMode(task.id === 'manual_other' ? 'manual' : 'fixed-cpu');
            }} />)}
        </Fragment>)}
        <Copy>{taskOption(inspectionTask)?.scope}</Copy>
        <Copy>{taskOption(inspectionTask)?.evaluation} · {photoRatesLabel(inspectionTask)}</Copy>
        <Copy>{aiMode === 'manual' ? '이 분야는 AI 판정 없이 직접 검토합니다.'
          : '기존 고정 모델의 연구 출력입니다. 평가가 부족하거나 개선이 필요한 분야도 포함하며, 최종 판단은 점검자가 합니다.'}</Copy>
      </> : <Copy>모의 결과는 실제 사진의 손상 판정이 아닙니다.</Copy>}
    </Card>
    <Card><PhotoField label="조치 전 사진" photo={photo} onChange={setPhoto} /></Card>
    <Card>
      <Field label="점검 위치" value={location} onChangeText={setLocation} />
      <Field label="발견 메모" value={description} onChangeText={setDescription} multiline />
      <Field label="수동 권장 조치" value={action} onChangeText={setAction} multiline />
    </Card>
    {!!error && <Copy tone="error">{error}</Copy>}
    <Copy>등록하면 미조치 상태로 저장되고 {aiMode === 'manual' ? '직접 검토' : 'AI 검토'}가 열립니다.</Copy>
    <Button title={aiMode === 'manual' ? '등록 후 직접 검토' : '등록 후 AI 검토'} onPress={register} />
  </Page>;
}

import { useState } from 'react';
import { router, useLocalSearchParams } from 'expo-router';
import { StyleSheet, View } from 'react-native';
import { Button, Card, Copy, message, Page } from '@/components/safelog-ui';
import { StatusBadge, workflowStatuses } from '@/features/workflow/status-badge';
import { roleLabels } from '@/features/auth/role-labels';
import { useFindings, useProfiles, useSession, workflow } from '@/features/auth/session';

export default function AssignedTasksPage() {
  const { user } = useSession();
  const records = useFindings();
  const profiles = useProfiles();
  const { filter } = useLocalSearchParams<{ filter?: string }>();
  const [error, setError] = useState('');
  if (!user) return null;
  const activeOnly = filter === 'active';
  const findings = activeOnly ? records.filter(finding => finding.status === 'InProgress') : records;

  function attempt(action: () => void) {
    try {
      setError('');
      action();
    } catch (error) {
      setError(message(error));
    }
  }

  return <Page title={activeOnly ? "진행 중 작업" : user.role === "Assignee" ? "배정된 작업" : "개선 조치 관리"}>
    <Copy>현재 역할 · {roleLabels[user.role]}</Copy>
    {!activeOnly && <Card>
      <Copy>점검 현황</Copy>
      {workflowStatuses.map(status => <View key={status} style={styles.statusRow}>
        <StatusBadge status={status} />
        <Copy>{records.filter(item => item.status === status).length}건</Copy>
      </View>)}
    </Card>}
    {!!error && <Copy tone="error">{error}</Copy>}
    {!findings.length && <Card><Copy>{activeOnly ? "진행 중인 작업이 없습니다." : "표시할 작업이 없습니다."}</Copy></Card>}
    {findings.map(finding => {
      const params = { id: finding.id };
      const assignee = profiles.find(person => person.id === finding.assigneeId);
      return <Card key={finding.id}>
        <Copy>{finding.location}</Copy>
        <StatusBadge status={finding.status} />
        <Copy>{finding.description}</Copy>
        <Copy tone={finding.risk && finding.risk >= 4 ? "risk" : "normal"}>위험 등급 · {finding.risk ?? "미지정"} / 5</Copy>
        <Copy>담당자 · {assignee?.name ?? '미배정'}</Copy>
        {user.role === 'Manager' && finding.status === 'Open' && !assignee && profiles.filter(person => person.role === 'Assignee').map(person =>
          <Button key={person.id} title={person.name + '에게 배정'} onPress={() => attempt(() => {
            workflow.execute(user, finding.id, { kind: 'assign', assigneeId: person.id });
          })} />)}
        {user.role === 'Assignee' && finding.status === 'Open' && <Button title="조치 시작" onPress={() => attempt(() => {
          workflow.execute(user, finding.id, { kind: 'start' });
          router.push({ pathname: '/action', params });
        })} />}
        {user.role === 'Assignee' && finding.status === 'InProgress' && <Button title="조치 내용 제출" onPress={() => router.push({ pathname: '/action', params })} />}
        {(finding.status === 'PendingReview' || finding.status === 'Verified') && <Button
          title={user.role === 'Inspector' ? '전후 비교 · 최종 확인' : '조치 전후 비교 보기'}
          onPress={() => router.push({ pathname: '/comparison', params })} />}
        {user.role === 'Inspector' && finding.status === 'Open' && <Button title="AI 위험 검토" secondary onPress={() => router.push({ pathname: '/hazard', params })} />}
        {user.role !== 'Assignee' && <Button title={finding.status === 'Verified' ? '완료 보고서' : '진행 보고서'} secondary onPress={() => router.push({ pathname: '/report', params })} />}
      </Card>;
    })}
  </Page>;
}

const styles = StyleSheet.create({
  statusRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
});

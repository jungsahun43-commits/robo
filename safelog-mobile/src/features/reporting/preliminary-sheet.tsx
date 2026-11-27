import { Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { checklistGroups, inspectionColumns, type Checklist, type ChecklistRow } from './checklist';

const widths = [72, 220, 55, 45, 52, 115, 56];
const tableWidth = widths.reduce((total, width) => total + width, 0);

export function PreliminarySheet({ info, checklist, onChange }: {
  info: readonly (readonly [string, string])[];
  checklist: Checklist;
  onChange: (id: string, update: ChecklistRow) => void;
}) {
  return <View style={styles.sheet}>
    <Text style={styles.caption}>설비·시설 사진 외관 점검표</Text>
    <Text accessibilityRole="header" style={styles.title}>작 업 장 순 회 점 검 표</Text>
    <View style={styles.info}>
      {info.map(([label, value]) => <View key={label} style={styles.row}>
        <Text style={styles.infoLabel}>{label}</Text>
        <Text style={styles.infoValue}>{value}</Text>
      </View>)}
    </View>
    <Text style={styles.hint}>점검표를 좌우로 움직여 결과와 조치사항을 입력하세요.</Text>
    <ScrollView horizontal keyboardShouldPersistTaps="handled">
      <View style={[styles.table, { width: tableWidth + 1 }]}>
        <View style={styles.row}>{inspectionColumns.map((label, index) =>
          <View key={label} style={[styles.cell, styles.heading, { width: widths[index] }]}><Text style={styles.center}>{label}</Text></View>)}</View>
        {checklistGroups.map(({ label, items }) => <View key={label}>
          <Text style={styles.group}>{label}</Text>
          {items.map(([id, category, description]) => {
            const row = checklist[id];
            return <View key={id} style={styles.row}>
              <View style={[styles.cell, { width: widths[0] }]}><Text>{category}</Text></View>
              <View style={[styles.cell, { width: widths[1] }]}><Text>{description}</Text></View>
              <Mark width={widths[2]} label={description + ' 해당 여부'} checked={row?.applicable === true}
                value={row?.applicable === false ? '해당 없음' : row?.suggestedByAi ? '☑ AI 제안' : undefined}
                onPress={() => onChange(id, row?.applicable === undefined ? { applicable: true } : row.applicable
                  ? { applicable: false, result: undefined, completed: undefined } : { applicable: undefined })} />
              <Mark width={widths[3]} label={description + ' 적정'} checked={row?.result === 'adequate'}
                disabled={row?.applicable === false} onPress={() => onChange(id, { applicable: true, result: row?.result === 'adequate' ? undefined : 'adequate' })} />
              <Mark width={widths[4]} label={description + ' 부적정'} checked={row?.result === 'inadequate'}
                disabled={row?.applicable === false} onPress={() => onChange(id, { applicable: true, result: row?.result === 'inadequate' ? undefined : 'inadequate' })} />
              <View style={[styles.cell, { width: widths[5] }]}>
                <TextInput accessibilityLabel={description + ' 조치사항'} multiline value={row?.note ?? ''}
                  onChangeText={note => onChange(id, { note })} style={styles.input} />
              </View>
              <Mark width={widths[6]} label={description + ' 조치 완료'} checked={row?.completed}
                disabled={row?.applicable === false} onPress={() => onChange(id, { completed: !row?.completed })} />
            </View>;
          })}
        </View>)}
      </View>
    </ScrollView>
  </View>;
}

function Mark({ width, label, checked, value, disabled, onPress }: {
  width: number; label: string; checked?: boolean; value?: string; disabled?: boolean; onPress: () => void;
}) {
  return <Pressable accessibilityRole="checkbox" accessibilityLabel={label} accessibilityValue={value ? { text: value } : undefined}
    accessibilityState={{ checked: !!checked, disabled }} disabled={disabled} onPress={onPress}
    style={[styles.cell, { width }]}>
    <Text style={styles.center}>{value ?? (checked ? '☑' : '□')}</Text>
  </Pressable>;
}

const styles = StyleSheet.create({
  sheet: { backgroundColor: 'white', padding: 10, borderWidth: 1, borderColor: '#222', gap: 8 },
  caption: { textAlign: 'center', fontSize: 11, color: '#18243a' },
  title: { textAlign: 'center', fontSize: 19, fontWeight: '700', color: '#18243a' },
  info: { borderTopWidth: 1, borderLeftWidth: 1, borderColor: '#333' },
  row: { flexDirection: 'row' },
  infoLabel: { width: '33%', padding: 6, backgroundColor: '#e1e7f3', borderBottomWidth: 1, borderRightWidth: 1, borderColor: '#333' },
  infoValue: { flex: 1, padding: 6, borderBottomWidth: 1, borderRightWidth: 1, borderColor: '#333' },
  hint: { fontSize: 12, color: '#52627A' },
  table: { borderTopWidth: 1, borderLeftWidth: 1, borderColor: '#333' },
  cell: { minHeight: 44, justifyContent: 'center', padding: 5, borderBottomWidth: 1, borderRightWidth: 1, borderColor: '#333' },
  heading: { backgroundColor: '#f4f6fa' },
  group: { backgroundColor: '#e1e7f3', padding: 5, fontWeight: '700', borderBottomWidth: 1, borderRightWidth: 1, borderColor: '#333' },
  center: { textAlign: 'center', fontSize: 12, color: '#18243a' },
  input: { color: '#18243a', fontSize: 12, minHeight: 34 },
});

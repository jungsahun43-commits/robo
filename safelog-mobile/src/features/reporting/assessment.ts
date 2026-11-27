export const assessmentFields = [
  { key: 'implementation', label: '개선대책 이행 결과' },
] as const;

export type Assessment = Partial<Record<(typeof assessmentFields)[number]['key'] | 'workers' | 'representative', string>>;

export function missingAssessmentFields(assessment: Assessment) {
  return assessmentFields.filter(({ key }) => !assessment[key]?.trim());
}

export function assessmentLines(assessment: Assessment) {
  return assessmentFields.map(({ key, label }) => label + ' · ' + (assessment[key]?.trim() || '미입력'));
}

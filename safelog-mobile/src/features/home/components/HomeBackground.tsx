import { Image } from 'expo-image';
import { StyleSheet } from 'react-native';

export function HomeBackground() {
  return <Image accessible={false} source={require('../../../../assets/safelog/hero.svg')} style={styles.art} />;
}

const styles = StyleSheet.create({
  art: { position: 'absolute', width: 151, height: 150, right: -12, top: 9, opacity: 0.68 },
});

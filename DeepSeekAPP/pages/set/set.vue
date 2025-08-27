<template>
	<!-- 绑定触摸事件：事件tap，绑定@，事件处理函数 -->
	<view @tap="editUsername">
		<!-- 文本插值 -->
		<view class="">姓名：{{student}}</view>
		<view class="">
			<image :src="avatar" mode=""></image>
		</view>
		<view class="">战队：{{team}}</view>

	</view>

</template>

<script>
	export default {
		// 数据一定要放在data里面
		data() {
			return {
				// 所有的数据都要放在这里
				student: '喻文州',
				team: '蓝雨',
				avatar: 'https://w.wallhaven.clbug.com/full/9o/wallhaven-9om6q8.jpg'
			}
		},
		// 事件处理函数一定要在methods里定义
		methods: {
			editUsername() {
				// 修改数据
				this.student = '索克萨尔'
				this.avatar = 'https://w.wallhaven.clbug.com/full/og/wallhaven-og3d99.jpg'
			}
		}
	}
</script>

<style>

</style>
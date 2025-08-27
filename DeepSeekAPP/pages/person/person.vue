<template>
	<view class="container">

		<!-- 账户 -->
		<view class="section">
			<view class="section-title">账户</view>
			<view class="list-item">
				<image class="item-icon" src="../../static/icons/phone.png"></image>
				<text class="item-text">手机号</text>
				<text class="item-value">123****7890</text>
			</view>
			<view class="list-item">
				<image class="item-icon" src="../../static/icons/wechat.png"></image>
				<text class="item-text">微信</text>
				<text class="item-value">已绑定</text>
			</view>
			<view class="list-item">
				<image class="item-icon" src="../../static/icons/data.png"></image>
				<text class="item-text">数据管理</text>
				<image class="arrow-icon" src="../../static/icons/arrow-right.png"></image>
			</view>
		</view>

		<!-- 应用 -->
		<view class="section">
			<view class="section-title">应用</view>
			<view class="list-item">
				<image class="item-icon" src="../../static/icons/language.png"></image>
				<text class="item-text">语言</text>
				<text class="item-value">中文（简体中文）</text>
				<image class="arrow-icon" src="../../static/icons/arrow-right.png"></image>
			</view>
			<view class="list-item">
				<image class="item-icon" src="../../static/icons/theme.png"></image>
				<text class="item-text">外观</text>
				<text class="item-value">系统</text>
				<image class="arrow-icon" src="../../static/icons/arrow-right.png"></image>
			</view>
			<view class="list-item">
				<image class="item-icon" src="../../static/icons/fontsize.png"></image>
				<text class="item-text">字体大小</text>
				<image class="arrow-icon" src="../../static/icons/arrow-right.png"></image>
			</view>
		</view>

		<!-- 关于 -->
		<view class="section">
			<view class="section-title">关于</view>
			<view class="list-item">
				<image class="item-icon" src="../../static/icons/update.png"></image>
				<text class="item-text">检查更新</text>
				<text class="item-value">2.3.3(88)</text>
				<image class="arrow-icon" src="../../static/icons/arrow-right.png"></image>
			</view>
			<view class="list-item">
				<image class="item-icon" src="../../static/icons/agreement.png"></image>
				<text class="item-text">服务协议</text>
				<image class="arrow-icon" src="../../static/icons/arrow-right.png"></image>
			</view>
		</view>

		<!-- 联系我们 -->
		<view class="section">
			<view class="list-item">
				<image class="item-icon" src="../../static/icons/contact.png"></image>
				<text class="item-text">联系我们</text>
				<image class="arrow-icon" src="../../static/icons/arrow-right.png"></image>
			</view>
		</view>
	</view>
</template>

<style>
	.container {
		background-color: #f8f8f8;
	}

	.nav-bar {
		display: flex;
		align-items: center;
		height: 100rpx;
		padding: 0 30rpx;
		background-color: #ffffff;
		border-bottom: 1rpx solid #eaeaea;
	}

	.icon-back {
		width: 40rpx;
		height: 40rpx;
	}

	.nav-title {
		flex: 1;
		text-align: center;
		font-size: 36rpx;
		font-weight: bold;
	}

	.section {
		margin-top: 20rpx;
		background-color: #ffffff;
		border-radius: 12rpx;
		overflow: hidden;
	}

	.section-title {
		padding: 20rpx 30rpx;
		font-size: 26rpx;
		color: #999999;
	}

	.list-item {
		display: flex;
		align-items: center;
		padding: 26rpx 30rpx;
		border-top: 1rpx solid #f0f0f0;
	}

	.list-item:first-child {
		border-top: none;
	}

	.item-icon {
		width: 40rpx;
		height: 40rpx;
		margin-right: 20rpx;
	}

	.item-text {
		flex: 1;
		font-size: 30rpx;
		color: #333333;
	}

	.item-value {
		font-size: 28rpx;
		color: #999999;
		margin-right: 20rpx;
	}

	.arrow-icon {
		width: 28rpx;
		height: 28rpx;
	}
</style>



<script>
	export default {
		data() {
			return {

			}
		},
		methods: {

		}
	}
</script>
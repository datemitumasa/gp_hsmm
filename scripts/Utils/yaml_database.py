#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yamlから読み込んだ物体情報、場所情報を管理するPythonファイルです。


"""
from __future__ import unicode_literals
import rospy

import yaml
import codecs
import copy
import rosparam

class DataBaseError(Exception):
    def __init__(self, message, *args):
        super(DataBaseError, self).__init__(message, args)
        self.message = message

    def __str__(self, ):
        print("")
        rospy.logwarn(self.message)
        return self.message

class UnitData(object):
    """
    単一のデータを表す基底クラス

    全てのデータはyamlから読み込んだ辞書．
    Data構造は以下のように

    e.g.)object.yaml
    - name: お茶
      id: 0
      pos: (1, 2, 3, 4)
    - name: マックスコーヒー
      id: 1
      pos: (2, 3, 4, 5)
    """

    def __init__(self, kind, file_name=None):
        """
        :param str filename: 読み込むDataファイル(yaml形式)
        """
        self.__kind = kind
        self.__data = []
        if file_name:
            self.loadFile(file_name)
        try:
            self.__data = rospy.get_param(self.__kind)
        except:
            pass
    def __str__(self,):
        contexts = []
        for i, data in enumerate(self.__data):
            contexts.append("===== {0} =====".format(i))
            for k, v in data.iteritems():
                contexts.append("{0}: {1}\t({2})".format(k, v, type(v)))
        context = "\n".join(contexts)
        return context.encode("sjis")

    def searchKey(self, from_value, from_key, to_key):
        """
        検索用内部メソッド
        :param from_value: 求めたい値
        :param str from_key: 検索に使うキー
        :param str to_key: 欲しいキー
        :return: 検索結果(見つからなかった-> None, 1つみつかった-> 結果, 複数見つかった-> 見つかった分の結果をリストで)
        Usage:
            # データベース上から`name`が`お茶`であるデータの`id`を検索する
            tea_id = _database.searchKey("お茶", "name", "id")
        """
        find_datas = []
        for data in self.__data:
            # マッチングしたいキーの値をもらう
            try:
                key = data[from_key]
            except KeyError:
                continue
            # マッチングしたいキーの値のチェック
            if key == from_value:
                try:
                    find_datas.append(data[to_key])
                except KeyError:
                    rospy.logwarn(u"not found {0}in {1} no data is here".format(from_value, to_key))

        if len(find_datas) == 1:
            return find_datas[0]
        elif len(find_datas) > 1:
            rospy.logwarn(u"{0}: {1} is not only one".format(from_key, from_value))
            return find_datas
        else:
            return -1            
    def loadFile(self, file_name):
        """
        指定されたfileを読み込むメソッド
        :param str file_name: 読み込むDataファイル(yaml形式)
        """
        # ファイル読み込み
        try:
#            data = codecs.open(file_name, "r", "sjis").read()
            data = open(file_name, "r")
            yaml_file = yaml.load(data)
            rosparam.upload_params("/", yaml_file)
            data.close()
        except IOError:
            rospy.logwarn("not found {0}".format(file_name))
            raise
        # yaml変換
        try:
            self.__data = rospy.get_param(self.__kind)
        except:
            rospy.logwarn("wrong file name {0}".format(file_name))
            raise

    def saveFile(self, file_name):
        """
        現在のデータをyamlで保存するメソッド
        :param str file_name: 書き出すファイル名(yaml形式)
        """
        with codecs.open(file_name, "w", "sjis") as f:
            yaml.safe_dump(self.__data, f, allow_unicode=True, default_flow_style=False)

    def name2id(self, name):
        """
        保存したデータのIDを名前から検索するメソッド
        :param str name: 検索したいデータの名前
        :rtype: int
        :return: IDがあった場合，そのIDを返す．(失敗した場合はNone)
        """
	#print name
        data = self.searchKey(name, "name", "id")
	#print data
	return data

    def name2pos(self, name):
        """
        データの名前から登録してある座標を取得するメソッド
        :param str name: 検索したいデータの名前
        :rtype: tuple of float
        :return: IDがあった場合，そのIDを返す．(失敗した場合はNone)
        """
        return self.searchKey(name, "name", "pos")

    def id2name(self, id):
        """
        idからDataの名前を検索するメソッド
        :param int id: ほしいデータのid
        :rtype: str
        :return: データの名前(失敗した場合はNone)
        """
        return self.searchKey(id, "id", "name")

    def id2pos(self, id):
        """
        idから対応するDataのpositionを検索する
        :param int id: 欲しいデータid
        :rtype: list or tuple
        :return: 場所の値(失敗した場合はNone)
        """
        return self.searchKey(id, "id", "pos")

    def pos2name(self, pos):
        """
        Dataのpositionから対応するDataの名前を検索する
        :param list or tuple of float pos: 検索に使うPositionデータ
        :rtype: str
        :return: 名前(失敗した場合はNone)
        """
        return self.searchKey(pos, "pos", "name")

    def pos2id(self, pos):
        """
        Dataのpositionから対応するDataの名前を検索する
        :param list or tuple of float pos: 検索に使うPositionデータ
        :rtype: int
        :return: 場所に対応するid(失敗した場合はNone)
        """
        return self.searchKey(pos, "pos", "id")

    def name2tf_base(self, name):
        """
        Dataのpositionから対応するDataの名前を検索する
        :param list or tuple of float pos: 検索に使うPositionデータ
        :rtype: int
        :return: 場所に対応するid(失敗した場合はNone)
        """
        return self.searchKey(name, "name", "tf_base")

    def id2tf_base(self, id):
        """
        Dataのpositionから対応するDataの名前を検索する
        :param list or tuple of float pos: 検索に使うPositionデータ
        :rtype: int
        :return: 場所に対応するid(失敗した場合はNone)
        """
        return self.searchKey(id, "name", "tf_base")


    def addData(self, name=None, id=None, pos=None, **kwargs):
        """
        データを追加するメソッド
        :param str name: データの名前
        :param int id: データid
        :param tuple pos: Positionデータ
        :param kwargs: その他に追加したいデータ
        """
        add_data = {"name": name,
                    "id": id,
                    "pos": pos}
        for k, v in kwargs.iteritems():
            add_data.update({k: v})
        self.__data.append(add_data)

    def deleteData(self, name=None, id=None, pos=None, **kwargs):
        """
        引数に与えられたデータの中にマッチするものがあれば削除するメソッド
        :param str name: データの名前
        :param int id: データid
        :param tuple pos: Positionデータ
        :param kwargs: その他に削除したいデータ
        """
        delete_data = []
        for i, data in enumerate(self.__data):
            try:
                # 引数が指定されていて，かつ，データベースとマッチしていれば削除リストに登録
                if (name is not None) and (name == data["name"]):
                    delete_data.append(data)
                    continue
                if (id is not None) and (id == data["id"]):
                    delete_data.append(data)
                    continue
                if (pos is not None) and (pos == data["pos"]):
                    delete_data.append(data)
                    continue
                for k, v in kwargs.iteritems():
                    if (kwargs is not None) and (v == data[k]):
                        delete_data.append(data)
                        continue
            except KeyError:
                pass

        # 削除
        for data in delete_data:
            self.__data.remove(data)

    def showData(self, ):
        """
        データの中身を表示するメソッド(Debug用)
        """
        print self

    def getAllData(self, ):
        """
        全てのデータを返すメソッド．デバッグなどに利用してください．
        :rtype: list of dict
        :return: 所持してあるデータがlistの各要素として返す

        Usage:
            data = _database.object.getAllData()
            data[0]["name"] # -> 最初に入っているデータのname要素にアクセス
        """
        return copy.deepcopy(self.__data)

class ObjectData(UnitData):
    def __init__(self, file_name=None):
        super(ObjectData, self).__init__(file_name)

    def category2name(self, category):
        """
        カテゴリー名からnameを検索するメソッド
        :param unicode category: カテゴリー名
        :rtype: unicode
        :return: name
        """
        return self.searchKey(category, "category", "name")

    def category2id(self, category):
        """
        カテゴリー名からidを検索するメソッド
        :param unicode category: カテゴリー名
        :rtype: int
        :return: id
        """
        return self.searchKey(category, "category", "id")

    def category2pos(self, category):
        """
        カテゴリー名からposを検索するメソッド
        :param unicode category: カテゴリー名
        :rtype: list of float
        :return: pos
        """
        return self.searchKey(category, "category", "pos")

    def name2category(self, name):
        """
        nameからcategoryを検索するメソッド
        :param unicode name: name
        :rtype: unicode
        :return: category
        """
        return self.searchKey(name, "name", "category")

    def id2category(self, id):
        """
        idからcategoryを検索するメソッド
        :param int name: id
        :rtype: unicode
        :return: category
        """
        return self.searchKey(id, "id", "category")

    def pos2category(self, pos):
        """
        idからcategoryを検索するメソッド
        :param int name: id
        :rtype: unicode
        :return: category
        """
        return self.searchKey(pos, "pos", "category")

class PoseData(UnitData):
    def __init__(self, file_name=None):
        super(PoseData, self).__init__(file_name)

    def name2pose(self, name):
        """
        Dataのpositionから対応するDataの名前を検索する
        :param string: 検索に使うPositionデータ
        :rtype: int
        :return: 姿勢に対応した関節角度(失敗した場合はNone)
        """
        return self.searchKey(name, "name", "joint_pose")


class FaceData(UnitData):
    def __init__(self, file_name=None):
        super(FaceData, self).__init__(file_name)

    def name2gender(self, name):
        """
        Dataのpositionから対応するDataの名前を検索する
        :param string: 検索に使うPositionデータ
        :rtype: int
        :return: 姿勢に対応した関節角度(失敗した場合はNone)
        """
        return self.searchKey(name, "name", "gender")


class CollisionData(UnitData):
    def __init__(self, file_name=None):
        super(CollisionData, self).__init__(file_name)

    def name2map_pos(self, name):
        """
        Dataのnameから対応するcollisionの座標を検索する
        :param string: 検索に使うPositionデータ
        :rtype: int
        :return: 姿勢に対応した関節角度(失敗した場合はNone)
        """
        return self.searchKey(name, "name", "map_pos")

    def name2option(self, name):
        """
        Dataのnameから対応するcollisionのoptionを検索する
        :param string: 検索に使うPositionデータ
        :rtype: int
        :return: 姿勢に対応した関節角度(失敗した場合はNone)
        """
        return self.searchKey(name, "name", "collision_option")

class SentenceData(UnitData):
    def __init__(self, file_name=None):
        super(SentenceData, self).__init__(file_name)
    def name2sentence(self, name):
        """
        Dataのnameから対応するcollisionの座標を検索する
        :param string: 検索に使うPositionデータ
        :rtype: int
        :return: 姿勢に対応した関節角度(失敗した場合はNone)
        """
        return self.searchKey(name, "name", "sentence")


class WordData(UnitData):
    def __init__(self, file_name=None):
        super(WordData, self).__init__(file_name)
    def exchange(self, word):
        """
        Dataのnameから対応するcollisionの座標を検索する
        :param string: 検索に使うPositionデータ
        :rtype: int
        :return: 姿勢に対応した関節角度(失敗した場合はNone)
        """
        data = self.getAllData()
        for dic in data:
            words = dic["words"]
            if word in words:
                return dic["name"]
        print("not dfine words")

        return ""


class DataBase(object):
    """
    データベースクラス(シングルトン)

    Usage:
        database = hsr_orders.Utils.DataBase(
                        object_file="object.yaml",
                        fece_file="face.yaml",
                        location_file="location.yaml")

        1) 設定ファイルを使ってデータを読み込む場合
            database.<data_type>.loadData(filename)

        2) データを追加したいとき
            database.<data_type>.addData(name, id, pos)

        3) データを上書き(変更)したいとき
            database.<data_type>.deleteData(name, id, pos)
            database.<data_type>.addData(name, id, pos)
    """

    object = ObjectData("object")
    face = FaceData("face")
    location = ObjectData("location")
    tf  =   ObjectData("tf_task_point")
    pose = PoseData("position")
    collision = CollisionData("collision")
    answer = ObjectData("sentence")
    word = WordData("wordbase")
    def __init__(self, object_file=None, face_file=None, location_file=None, tf_file=None):
        """
        :param str object_file: 物体情報を記したyamlファイル
        :param str face_file: 顔情報を記したyamlファイル
        :param str location_file: 場所情報を記したyamlファイル
        :param str tf_file: TF情報を記したyamlファイル
        """
        if object_file != None:
            self.object.loadFile(object_file)
        if face_file != None:
            self.face.loadFile(face_file)
        if location_file != None:
            self.location.loadFile(location_file)
        if tf_file != None:
            self.tf.loadFile(tf_file)
